"""The scripted operator for the film's human-in-the-loop shots (P1-B, P1-D in media/film/SCRIPT.md).

Runs beside `demo.py --film ...` on the same Windows machine. It watches the operator page's /api/pending, saves every
question it sees with its wall time (and the frame the page showed), and answers each one as the plan says: an option
after so many seconds, the model's own pick, or nothing at all (the demo then times out to the safe answer). It can
also stop d3 27B partway, so the questions after that climb straight to the page. Standard library only; the page is
redrawn from what it saved by render_operator_page.py, so no browser has to run beside the sim.

    .\\run.ps1 python scripts/film_answerer.py --out output/film/p1b --plan "*=model@4"
    .\\run.ps1 python scripts/film_answerer.py --out output/film/p1d --stop-d3-after 40 --plan "1=defective@5,2=none"

Plan: comma-separated `<key>=<what>@<seconds>`. Keys: `<n>` (the page's n-th question, from 1), `s<k>` (the k-th
question after d3 was stopped), `s*` (any other question after the stop), `*` (any other question). <what> is an
option, `model` (the model's pick, read from the "climbed" line), `truth` (the simulator's answer, which
`demo.py --film` adds to the page's data so the operator answers as a person seeing the scene would), `pick:a|b|c`
(the first of these the question offers) or `none`. Unlisted questions are not answered.

    --stop-d3-after 60 --plan "*=model@4,s1=pick:defective|grasp|yes@5,s2=none,s*=pick:defective|grasp|yes@3"
Writes <out>/page.jsonl (one row per state change) and <out>/page/q<id>.png.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_plan(text: str) -> dict[str, tuple[str, float]]:
    """'1=defective@4,*=none' -> {'1': ('defective', 4.0), '*': ('none', 0.0)}."""
    plan: dict[str, tuple[str, float]] = {}
    for item in filter(None, (p.strip() for p in text.split(","))):
        key, _, rest = item.partition("=")
        what, _, delay = rest.partition("@")
        valid_key = key in ("*", "s*") or key.isdigit() or (key.startswith("s") and key[1:].isdigit())
        if not valid_key or not what or (what.startswith("pick:") and len(what) == 5):
            raise ValueError(f"bad plan item {item!r}: want <n|s<k>|s*|*>=<option|model|pick:a|b|none>@<seconds>")
        plan[key] = (what, float(delay or 0))
    return plan


def model_pick(climbed: str, options: list[str]) -> str | None:
    """The option the last model before the page chose, from its "climbed" line (e.g. 'd3-27b-vision defective
    0.60', or 'decider-2b yes 0.55, gemma4:e4b no 0.62' on a longer ladder: the last rung is the one just below)."""
    words = climbed.replace(",", " ").split()
    for w in reversed(words):
        if w in options:
            return w
    return None


def decide(plan: dict[str, tuple[str, float]], n: int, options: list[str], climbed: str, since_stop: int | None = None, truth: str | None = None) -> tuple[str | None, float]:
    """What to answer to the n-th question (None: leave it) and after how many seconds. since_stop counts the
    questions since d3 was stopped (None before)."""
    keys = [str(n)] + ([f"s{since_stop}", "s*"] if since_stop is not None else []) + ["*"]
    what, delay = next((plan[k] for k in keys if k in plan), ("none", 0.0))
    if what == "none":
        return None, delay
    if what == "model":
        return model_pick(climbed, options), delay
    if what == "truth":
        return (truth if truth in options else None), delay
    if what.startswith("pick:"):
        return next((o for o in what[5:].split("|") if o in options), None), delay
    if what not in options:
        raise ValueError(f"question {n}: {what!r} is not one of {options}")
    return what, delay


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8790")
    p.add_argument("--out", required=True)
    p.add_argument("--plan", default="*=none")
    p.add_argument("--stop-d3-after", type=float, default=None, help="seconds after the page first answers; d3 27B is stopped then")
    p.add_argument("--quit-after", type=float, default=1800.0, help="give up after this many seconds")
    a = p.parse_args()
    plan = parse_plan(a.plan)
    out = Path(a.out)
    if (out / "page.jsonl").exists():  # page ids restart at 0 every demo: two takes in one log would mix questions
        raise SystemExit(f"{out} already holds a take: use a new folder")
    (out / "page").mkdir(parents=True, exist_ok=True)
    log = (out / "page.jsonl").open("w", encoding="utf-8")

    def note(row: dict) -> None:
        row = {"wall_t": round(time.time(), 3), **row}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    t_up = None
    t_last_up = time.time()
    begun = time.time()
    seen: dict[int, dict] = {}  # page id -> {n, due, answer}
    last_id = None
    d3_stopped = False
    since_stop = 0
    while time.time() - begun < a.quit_after:
        try:
            pending = json.loads(urlopen(a.url + "/api/pending", timeout=5).read() or b"{}")
        except OSError:
            if t_up is not None and time.time() - t_last_up > 10:  # the demo has closed its page: the run is over
                note({"event": "page_down"})
                break
            time.sleep(0.5)
            continue
        t_last_up = time.time()
        if t_up is None:
            t_up = time.time()
            note({"event": "page_up"})
        if a.stop_d3_after is not None and not d3_stopped and time.time() - t_up >= a.stop_d3_after:
            from handoff.servers import stop_d3

            d3_stopped = True
            note({"event": "stop_d3", "stopped": bool(stop_d3(say=lambda s: None))})
        qid = pending.get("id")
        if qid != last_id:
            if last_id is not None and last_id in seen and not seen[last_id].get("answered"):
                note({"event": "gone", "id": last_id})  # timed out to the safe answer, or answered elsewhere
            if qid is not None and qid not in seen:
                n = len(seen) + 1
                if d3_stopped:
                    since_stop += 1
                answer, delay = decide(plan, n, pending.get("options", []), pending.get("climbed", ""), since_stop if d3_stopped else None, pending.get("film_truth"))
                seen[qid] = {"n": n, "due": time.time() + delay, "answer": answer}
                if pending.get("frame"):
                    (out / "page" / f"q{qid}.png").write_bytes(base64.b64decode(pending["frame"]))
                note({"event": "question", "id": qid, "n": n, "text": pending.get("text"), "options": pending.get("options"), "criteria": pending.get("criteria"),
                      "climbed": pending.get("climbed"), "state_text": pending.get("state_text"), "film_truth": pending.get("film_truth"), "plan": answer, "after_s": delay})
            last_id = qid
        if qid is not None:
            s = seen[qid]
            if s["answer"] is not None and not s.get("answered") and time.time() >= s["due"]:
                body = json.dumps({"id": qid, "answer": s["answer"]}).encode()
                urlopen(Request(a.url + "/api/answer", data=body, method="POST"), timeout=5).read()
                s["answered"] = True
                note({"event": "answered", "id": qid, "answer": s["answer"]})
        time.sleep(0.2)
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
