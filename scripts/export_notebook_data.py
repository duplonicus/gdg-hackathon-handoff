"""Write the small data extract the public notebook reads (notebooks/decision_ladder_results.ipynb).

One row per labelled test04 decision, in log order, with only what the notebook scores: the question, the truth,
d3 27B's top answer and its probability, whether a hard rule answered it while the log was made (and with what), the
episode's fault type, and for in_bin decisions where the part actually was. No frames, no state text, no host names,
no absolute paths. Truths are loaded exactly as freeze_ladder.py confirm loads them (``benchmark.load`` relabels the
bin questions with the current geometry check; ``collapse_finish`` folds the old finish labels).

    .venv\\Scripts\\python.exe scripts\\export_notebook_data.py
    .venv\\Scripts\\python.exe scripts\\export_notebook_data.py --log output\\sort\\test04 --frozen output\\sort\\test03\\frozen-ladder.json

Writes notebooks/data/test04_decisions.csv and notebooks/data/frozen_ladder.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from benchmark import load, load_arms
from handoff_breakdown import where_part_was

from handoff.benchmark import collapse_finish, top

COLUMNS = [
    "decision",
    "episode",
    "fault",
    "question",
    "truth",
    "d3_answer",
    "d3_confidence",
    "rule_decided",
    "rule_answer",
    "part_was",
]


def rows(log: Path, arm: str) -> list[dict]:
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    answers = arms[arm]
    fault = {}
    for line in (log / "episodes.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        fault[f"ep{e['seed']:04d}"] = e.get("fault") or "clean"
    out = []
    for d in decisions:
        if d.get("truth") is None:
            continue
        a = answers.get(int(d["index_global"]))
        if a is None:
            raise SystemExit(
                f"{arm} has no answer for decision {d['index_global']}: the extract needs every labelled decision"
            )
        choice, conf = top(a["probs"])
        rule = d.get("answered_by") == "rule"
        out.append(
            {
                "decision": int(d["index_global"]),
                "episode": d["episode"],
                "fault": fault[d["episode"]],
                "question": d["question"],
                "truth": d["truth"],
                "d3_answer": choice,
                "d3_confidence": repr(float(conf)),  # full precision: the 0.80 cut is compared exactly
                "rule_decided": int(rule),
                "rule_answer": d["answer"] if rule else "",
                "part_was": where_part_was(d) if d["question"] == "in_bin" else "",
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", default=str(ROOT / "output" / "sort" / "test04"))
    p.add_argument("--frozen", default=str(ROOT / "output" / "sort" / "test03" / "frozen-ladder.json"))
    p.add_argument("--out", default=str(ROOT / "notebooks" / "data"))
    a = p.parse_args()
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    (arm,) = frozen["rungs"]  # the frozen ladder has one rung; the extract carries that arm's answers only
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = rows(Path(a.log), arm)
    with (out / "test04_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    summary = {
        "source": "frozen ladder of docs/PREREG.md (scripts/freeze_ladder.py freeze on all of test03)",
        "frozen_at_commit": frozen["run"]["git_commit"],
        "rungs": frozen["rungs"],
        "thresholds": frozen["thresholds"],
        "fit": frozen["fit"],
        "human_cost": frozen["human_cost"],
        "test03_prediction": frozen["prediction"],
    }
    (out / "frozen_ladder.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{len(data)} decisions, {len({r['episode'] for r in data})} episodes -> {out / 'test04_decisions.csv'}")
    print(f"frozen ladder ({arm} at {frozen['thresholds'][arm]}) -> {out / 'frozen_ladder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
