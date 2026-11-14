r"""Drives the real operator page in a real browser: a fake mic plays a clip, Whistle and d3 run for real, a person confirms.

Needs d3 27B on port 8774, Whistle (scripts/get_whistle.ps1) and Edge. The clip is Windows' speech voice, not a person,
so this proves the path works, not how well a real voice in a real room is heard.

    .\run.ps1 --with playwright python scripts/probe_voice_page.py D:\Models\whistle\samples\wrong-tray.wav no
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.sync_api import sync_playwright

from handoff.decider import QUESTIONS
from handoff.operator import HumanRung, OperatorPage
from handoff.rungs import DeciderRung
from handoff.servers import D3_URL
from handoff.voice import VoiceInput, Whistle


def main() -> int:
    clip, expected = sys.argv[1], sys.argv[2]
    question = QUESTIONS[sys.argv[3] if len(sys.argv) > 3 else "in_bin"]
    d3 = DeciderRung("d3-27b-vision", D3_URL, with_image=True, data_url=True, timeout=120.0, expect=("d3-27b",))
    print("served:", d3.check_model())
    page_server = OperatorPage(port=0, voice=VoiceInput(Whistle(), d3))
    page_server.start()
    result: dict = {}
    rung = HumanRung(page_server, idle=lambda: None, timeout_s=90.0, climbed=lambda: "d3-27b-vision 0.61")
    asker = threading.Thread(target=lambda: result.update(answer=rung.answer(question, "state text for the operator", None)), daemon=True)
    asker.start()
    errors: list[str] = []
    out = Path("output/voice_probe")
    out.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as play:
            browser = play.chromium.launch(channel="msedge", headless=True, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={Path(clip).as_posix()}%noloop", "--autoplay-policy=no-user-gesture-required",
            ])
            page = browser.new_context(viewport={"width": 1000, "height": 760}).new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
            page.goto(page_server.url)
            page.wait_for_selector("#speak", state="visible", timeout=10_000)
            page.click("#speak")
            page.wait_for_selector("#speak.on", timeout=10_000)
            print("recording:", page.inner_text("#speak"))
            time.sleep(float(sys.argv[4]) if len(sys.argv) > 4 else 5.0)
            page.click("#speak")
            started = time.monotonic()
            page.wait_for_function("document.getElementById('heard').textContent && !document.getElementById('heard').textContent.startsWith('Working')", timeout=60_000)
            print(f"heard after {time.monotonic() - started:.2f} s:", page.inner_text("#heard"))
            page.screenshot(path=str(out / f"page-{Path(clip).stem}.png"))
            print("still open before confirm:", asker.is_alive())
            if page.query_selector("#confirm-yes") is None:
                print("no option proposed")
                return 1
            print("button:", page.inner_text("#confirm-yes"))
            page.click("#confirm-yes")
            asker.join(10)
            browser.close()
    finally:
        page_server.close()
    answer = result.get("answer")
    print("answer:", answer.top if answer else None, "| extra:", answer.extra if answer else None, "| page errors:", errors)
    return 0 if answer and answer.top == expected and answer.extra.get("spoken") and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
