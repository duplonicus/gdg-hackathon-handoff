"""Turn a Kaggle Benchmarks run of scripts/kaggle_task.py into arms.jsonl rows, so the arm scores like any other.

    scripts/kaggle.sh b t download handoff-decision -m gemini-4-argon -o output/kaggle/runs
    bash run.sh scripts/kaggle_import.py --runs output/kaggle/runs/handoff-decision/<version>/gemini-4-argon --arm argon-vision

Each run file carries one decision's answer and stated confidence; the confidence goes to the chosen option and
the rest is spread evenly, as for the local self-confidence arms. Rows are appended to output/sort/<log>/arms.jsonl
for the log named in the row; decisions this arm already has are skipped, so a rerun is harmless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handoff.decider import QUESTIONS  # noqa: E402

SORT = Path(__file__).resolve().parents[1] / "output" / "sort"


def row_from_result(result: dict, arm: str, model: str) -> dict:
    options = QUESTIONS[result["question"]].options
    answer, conf = str(result["answer"]).strip().lower(), float(min(max(result["confidence"], 0.0), 1.0))
    if answer in options:
        rest = (1.0 - conf) / max(len(options) - 1, 1)
        probs, error = {o: (conf if o == answer else rest) for o in options}, None
    else:  # an answer outside the options is a failed rung, exactly as for the local arms
        probs, error = {o: 1.0 / len(options) for o in options}, f"answer {answer!r} is not an option"
    return {"arm": arm, "index_global": int(result["index_global"]), "question": result["question"], "probs": probs, "latency_ms": None,
            "error": error, "truth": result["truth"], "host": "Kaggle Benchmarks", "server": f"kaggle model proxy: {model}", "extra": {"self_confidence": conf}}


def results(run_dir: Path) -> list[tuple[dict, str]]:
    out = []
    for path in sorted(run_dir.rglob("*.run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        model = run.get("modelVersion", {}).get("slug", "")
        for r in run.get("results", []):
            if r.get("dictResult"):
                out.append((r["dictResult"], model))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--sort-dir", default=str(SORT))
    args = parser.parse_args()
    by_log: dict[str, list[dict]] = {}
    for result, model in results(Path(args.runs)):
        by_log.setdefault(result["log"], []).append(row_from_result(result, args.arm, model))
    for log, rows in by_log.items():
        out = Path(args.sort_dir) / log / "arms.jsonl"
        have = {int(json.loads(line)["index_global"]) for line in out.open(encoding="utf-8") if json.loads(line)["arm"] == args.arm} if out.exists() else set()
        new = [r for r in rows if r["index_global"] not in have]
        with out.open("a", encoding="utf-8") as handle:
            for r in sorted(new, key=lambda r: r["index_global"]):
                handle.write(json.dumps(r) + "\n")
        right = sum(1 for r in rows if r["error"] is None and max(r["probs"], key=r["probs"].get) == r["truth"])
        print(f"{log}: {len(rows)} answers, {len(new)} new rows for {args.arm} in {out} ({right} right, {sum(1 for r in rows if r['error'])} invalid)")


if __name__ == "__main__":
    main()
