"""Redraw the operator page for the film from what film_answerer.py saved (P1-B, P1-D in media/film/SCRIPT.md).

The real page (handoff.operator.PAGE, read from the source file, not retyped) is loaded in headless Chromium with its
/api/pending answered from page.jsonl, so each screenshot is the page exactly as it looked for that state. For each
question: the page with it waiting, then (if it was answered) the chosen button pressed, then the idle page again.
Runs in WSL with any Playwright; nothing here needs the sim or the GPU.

    PLAYWRIGHT_BROWSERS_PATH=... python render_operator_page.py output/film/p1b

Writes <dir>/page_shots/NNN_<state>.png and <dir>/page_shots.jsonl ({shot, wall_t, state, id}).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def page_html() -> str:
    """The PAGE string from src/handoff/operator.py, as the demo serves it."""
    src = (ROOT / "src" / "handoff" / "operator.py").read_text(encoding="utf-8")
    m = re.search(r'^PAGE = """(.*?)"""', src, re.S | re.M)
    if not m:
        raise SystemExit("PAGE not found in src/handoff/operator.py")
    return m.group(1)


def states(rows: list[dict]) -> list[dict]:
    """The page's states in wall-clock order: idle at the start, each question, its press, idle after."""
    out = []
    first = next((r["wall_t"] for r in rows), 0.0)
    out.append({"state": "idle", "wall_t": first, "id": None})
    asked: dict = {}
    for r in rows:
        if r["event"] == "question":
            asked[r["id"]] = r
            out.append({"state": "question", "wall_t": r["wall_t"], "id": r["id"], "row": r})
        elif r["event"] == "answered":
            q = asked.get(r["id"])
            # the press shows just before the answer went in, but never before its question appeared (a 0 s plan)
            at = max(r["wall_t"] - 0.4, q["wall_t"] + 0.05) if q else r["wall_t"] - 0.4
            out.append({"state": "press", "wall_t": at, "id": r["id"], "answer": r["answer"], "row": q})
            out.append({"state": "idle", "wall_t": r["wall_t"] + 0.3, "id": None})
        elif r["event"] == "gone":
            out.append({"state": "idle", "wall_t": r["wall_t"], "id": None})
    return sorted(out, key=lambda s: s["wall_t"])  # stable: a question and its press at one time keep that order


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=700)  # the page's content ends near 640 px
    a = p.parse_args()
    d = Path(a.dir)
    rows = [json.loads(line) for line in (d / "page.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shots = d / "page_shots"
    shots.mkdir(exist_ok=True)
    html = page_html()

    from playwright.sync_api import sync_playwright

    pending: dict = {}
    index = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": a.width, "height": a.height})
        page.route("**/api/pending", lambda route: route.fulfill(content_type="application/json", body=json.dumps(pending)))
        page.route("**/api/answer", lambda route: route.fulfill(content_type="application/json", body="{}"))
        page.route("http://operator.local/", lambda route: route.fulfill(content_type="text/html", body=html))
        page.goto("http://operator.local/")
        for k, s in enumerate(states(rows)):
            pending.clear()
            if s["state"] in ("question", "press"):
                q = s["row"]
                img = d / "page" / f"q{q['id']}.png"
                pending.update({"id": q["id"], "text": q["text"], "options": q["options"], "criteria": q.get("criteria") or {o: "" for o in q["options"]},
                                "state_text": q.get("state_text") or "", "climbed": q.get("climbed") or "", "voice": False,
                                "frame": base64.b64encode(img.read_bytes()).decode() if img.exists() else None})
            page.wait_for_timeout(700)  # the page polls every 500 ms
            if s["state"] == "press":
                # each button reads "<option> - <criteria>": match the option at the start, since a criteria text can
                # hold another option's word (sort_complete's "yes - no parts remain ...")
                button = page.locator("#opts button").filter(has_text=re.compile("^" + re.escape(s["answer"]) + " - ")).first
                button.evaluate("b => { b.style.background = '#2a6'; b.style.borderColor = '#6e9'; }")
            name = f"{k:03d}_{s['state']}.png"
            page.screenshot(path=str(shots / name))
            index.append({"shot": name, "wall_t": s["wall_t"], "state": s["state"], "id": s["id"]})
        browser.close()
    (d / "page_shots.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n", encoding="utf-8")
    print(f"{len(index)} page shots in {shots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
