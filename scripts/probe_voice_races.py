r"""Clicks the operator page's mic button at awkward moments in a real browser, with a slow mic permission and no models.

Exits non-zero if a microphone is left open or a recording starts that nobody asked for. Needs Edge.

    .\run.ps1 --with playwright python scripts/probe_voice_races.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.sync_api import sync_playwright

from handoff.decider import QUESTIONS
from handoff.operator import OperatorPage
from handoff.voice import VoiceInput

SLOW_MIC = """
const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
window.__streams = [];
navigator.mediaDevices.getUserMedia = async (c) => {
  await new Promise(r => setTimeout(r, 400));
  const s = await real(c); window.__streams.push(s); return s;
};
window.__live = () => window.__streams.filter(s => s.getTracks().some(t => t.readyState === 'live')).length;
"""


class Canned:
    def hear(self, question, wav):
        return {"transcript": "no", "option": "no", "confidence": 0.99, "reason": ""}


def open_question(page_server):
    threading.Thread(target=lambda: page_server.ask(QUESTIONS["in_bin"], "state", None, "", lambda: None, 60.0), daemon=True).start()


def main() -> int:
    failures = []
    voice: VoiceInput = Canned()  # type: ignore[assignment]
    with sync_playwright() as play:
        browser = play.chromium.launch(channel="msedge", headless=True, args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])

        server = OperatorPage(port=0, voice=voice)
        server.start()
        open_question(server)
        page = browser.new_page()
        page.add_init_script(SLOW_MIC)
        page.goto(server.url)
        page.wait_for_selector("#speak", state="visible")
        page.click("#speak")
        page.click("#speak")  # a second click while the permission is still pending
        page.wait_for_selector("#speak.on", timeout=5000)
        time.sleep(0.8)
        live = page.evaluate("window.__live()")
        print("double click: live microphones while recording =", live)
        if live != 1:
            failures.append(f"double click left {live} microphones open")
        page.click("#speak")
        page.wait_for_selector("#confirm-yes", timeout=5000)
        time.sleep(0.3)
        live = page.evaluate("window.__live()")
        print("after stopping: live microphones =", live)
        if live != 0:
            failures.append(f"{live} microphones still open after stopping")
        server.close()

        server = OperatorPage(port=0, voice=voice)
        server.start()
        open_question(server)
        page = browser.new_page()
        page.add_init_script(SLOW_MIC)
        page.goto(server.url)
        page.wait_for_selector("#speak", state="visible")
        page.click("#speak")
        page.click("#opts button")  # a button answers while the permission is still pending
        time.sleep(1.2)
        live, heard = page.evaluate("window.__live()"), page.text_content("#heard") or ""
        print(f"button during permission: live microphones = {live}, message = {heard!r}")
        if live != 0:
            failures.append(f"a button answer left {live} microphones open")
        if "Cannot read" in heard:
            failures.append(f"a button answer showed an error: {heard}")
        server.close()

        server = OperatorPage(port=0, voice=voice)
        server.start()
        open_question(server)
        page = browser.new_page()
        page.add_init_script(SLOW_MIC)
        page.goto(server.url)
        page.wait_for_selector("#speak", state="visible")
        page.evaluate("(async () => { await answer(current.id, current.options[0], false); await toggleSpeak(); })()")  # before the next poll hides the panel
        time.sleep(0.8)
        live, heard = page.evaluate("window.__live()"), page.text_content("#heard") or ""
        print(f"click just after answering: live microphones = {live}, message = {heard!r}")
        if live != 0:
            failures.append(f"a click just after answering left {live} microphones open")
        if "Cannot read" in heard:
            failures.append(f"a click just after answering showed an error: {heard}")
        server.close()
        browser.close()
    print("FAILED: " + "; ".join(failures) if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
