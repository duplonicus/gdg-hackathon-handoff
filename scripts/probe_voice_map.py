r"""How well does d3 read a typed operator reply as one of a question's options? Needs d3 27B up on port 8774.

The phrases were written before any answer was seen. They are one author's guess at how people talk, not recordings.
What counts most is "wrong": a confident proposal of the wrong option, which only the person's confirm then catches.

    .\run.ps1 python scripts/probe_voice_map.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handoff.decider import QUESTIONS
from handoff.rungs import DeciderRung
from handoff.servers import D3_URL
from handoff.voice import map_speech

# (question, what was said, the option it means; None = it picks no option, so asking again is the right result)
PHRASES = [
    ("in_bin", "yes", "yes"),
    ("in_bin", "yeah", "yes"),
    ("in_bin", "yep it's in there", "yes"),
    ("in_bin", "looks good, it landed fine", "yes"),
    ("in_bin", "that's in the right tray", "yes"),
    ("in_bin", "no", "no"),
    ("in_bin", "nope", "no"),
    ("in_bin", "no that's in the wrong tray", "no"),
    ("in_bin", "it fell on the floor", "no"),
    ("in_bin", "it missed", "no"),
    ("in_bin", "it's still in the hand", "no"),
    ("in_bin", "that went in the green one but it should be orange", "no"),
    ("in_bin", "hold on", None),
    ("in_bin", "I can't tell from this angle", None),
    ("in_bin", "what did you ask", None),
    ("in_bin", "yes no wait", None),
    ("finish", "good", "good"),
    ("finish", "looks fine to me", "good"),
    ("finish", "the paint's clean", "good"),
    ("finish", "that one's okay", "good"),
    ("finish", "defective", "defective"),
    ("finish", "it's scratched", "defective"),
    ("finish", "there's a bare patch on it", "defective"),
    ("finish", "send it to rework", "defective"),
    ("finish", "bad part", "defective"),
    ("finish", "um", None),
    ("finish", "show me another angle", None),
    ("safe_to_grasp", "go ahead", "grasp"),
    ("safe_to_grasp", "grab it", "grasp"),
    ("safe_to_grasp", "yeah pick it up", "grasp"),
    ("safe_to_grasp", "look again", "re_look"),
    ("safe_to_grasp", "I can't see it, take another look", "re_look"),
    ("safe_to_grasp", "let someone else decide", "ask_operator"),
    ("safe_to_grasp", "stop", "abort"),
    ("safe_to_grasp", "abort abort", "abort"),
    ("safe_to_grasp", "don't touch it, something's wrong", "abort"),
    ("safe_to_grasp", "what's the weather like", None),
    ("sort_complete", "yes it's done", "yes"),
    ("sort_complete", "all finished", "yes"),
    ("sort_complete", "no there's one left on the table", "no"),
    ("sort_complete", "not yet", "no"),
]


def grade(expected: str | None, proposed: str | None) -> str:
    if proposed is None:
        return "asked again" if expected is not None else "right"
    return "right" if proposed == expected else "WRONG"


def main() -> int:
    rung = DeciderRung("d3-27b-vision", D3_URL, with_image=True, data_url=True, timeout=120.0, expect=("d3-27b",))
    print("served:", rung.check_model())
    rows, tally = [], Counter()
    for question_id, said, expected in PHRASES:
        heard = map_speech(QUESTIONS[question_id], said, rung)
        verdict = grade(expected, heard["option"])
        group = "picks an option" if expected else "picks none"
        tally[(group, verdict)] += 1
        rows.append({"question": question_id, "said": said, "expected": expected, **heard, "verdict": verdict})
        print(f"{verdict:12s} {question_id:14s} {said!r:55s} -> {heard['option']} {heard['confidence']:.2f} {heard['reason']}", flush=True)
    print({f"{g}: {v}": n for (g, v), n in sorted(tally.items())})
    out = Path("output/voice_probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "map.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
