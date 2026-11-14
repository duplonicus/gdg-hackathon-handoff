"""Write the Phase 2 data extract the public notebook reads (notebooks/decision_ladder_results.ipynb, Phase 2 cells).

One row per labelled test05 decision, in log order: the shift, the question, the truth, and each replayed arm's top
answer and its probability. No frames, no state text, no host names, no paths. Truths are read as written in
decisions.jsonl (the capsule questions are not relabelled at load). An errored answer is left out and its retry used,
as capsule_arm_report.py and freeze_ladder.py do. Also copies the frozen sel03 ladder (thresholds per question).

    python3 scripts/export_capsule_notebook_data.py --log .claude/worktrees/capsules/output/capsules/test05 \\
        --frozen .claude/worktrees/capsules/output/capsules/sel03/frozen-ladder.json

Writes notebooks/data/test05_decisions.csv and notebooks/data/frozen_ladder_capsules.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("intern-4b-vision", "d3-27b-vision", "qwen-27b-vision", "qwen-27b-self-vision")


def rows(log: Path) -> list[dict]:
    decisions = [json.loads(line) for line in (log / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    answers: dict[str, dict[int, tuple[str, float]]] = {a: {} for a in ARMS}
    for line in (log / "arms.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r["arm"] not in answers:
            continue
        top = max(r["probs"], key=r["probs"].get)
        answers[r["arm"]][int(r["index_global"])] = (top, r["probs"][top])
    out = []
    for i, d in enumerate(decisions):
        if d.get("truth") is None:
            continue
        row = {"index": i, "shift": d["episode"], "question": d["question"], "truth": d["truth"]}
        for a in ARMS:
            ans = answers[a].get(i)
            row[f"{a}_answer"] = ans[0] if ans else ""
            row[f"{a}_p"] = round(ans[1], 6) if ans else ""
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    p.add_argument("--frozen", required=True)
    a = p.parse_args()
    data = rows(Path(a.log))
    dest = ROOT / "notebooks" / "data"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "test05_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    frozen = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    keep = {k: frozen[k] for k in ("rungs", "fit", "human_cost", "thresholds", "prediction")}
    keep["source"] = "sel03 (output/capsules/sel03/frozen-ladder.json); docs/PREREG_CAPSULES.md, committed 0981889 before test05 existed"
    (dest / "frozen_ladder_capsules.json").write_text(json.dumps(keep, indent=1), encoding="utf-8")
    missing = {arm: sum(1 for r in data if r[f"{arm}_answer"] == "") for arm in ARMS}
    print(f"{len(data)} decisions -> {dest / 'test05_decisions.csv'}; unanswered per arm: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
