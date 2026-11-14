"""Freeze the picked ladder for test04, then score test04 with it once (docs/PREREG.md).

freeze: fit the ladder chosen by pick_ladder.py on ALL of test03, with the fit it was picked with, and write the
thresholds as numbers. These go into docs/PREREG.md, which is committed before test04 is replayed.

    .venv\\Scripts\\python.exe scripts\\freeze_ladder.py freeze --log output\\sort\\test03

confirm: score test04 with the frozen thresholds exactly as written (no refit) and report the four PREREG rates with
95% by-episode intervals, overall and on the judgement questions, beside the test03 ten-split prediction.

    .venv\\Scripts\\python.exe scripts\\freeze_ladder.py confirm --log output\\sort\\test04 --frozen output\\sort\\test03\\frozen-ladder.json

Writes <test03>/frozen-ladder.json and <test04>/confirmation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import load, load_arms, run_info

from handoff.benchmark import (
    choose_thresholds,
    choose_thresholds_per_question,
    collapse_finish,
    compose_ladder,
    score,
)

RULE_DECIDED = "safe_to_grasp"


def fit(name: str, min_decisions: int):
    """The fits pick_ladder.py compares, by the names it writes into ladder-pick.json."""
    if name == "per rung":
        return choose_thresholds
    if name == "per question":
        return choose_thresholds_per_question
    if name.startswith("per question, under"):
        return lambda d, r, **kw: choose_thresholds_per_question(d, r, min_decisions=min_decisions, **kw)
    raise SystemExit(f"unknown fit {name!r}")


def freeze(log: Path) -> None:
    pick_report = json.loads((log / "ladder-pick.json").read_text(encoding="utf-8"))
    pick, run = pick_report["pick"], pick_report["run"]
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    rungs = pick["rungs"]
    thresholds = fit(pick["fit"], run["min_decisions"])(decisions, [(r, arms[r]) for r in rungs], human_cost=run["human_cost"])
    frozen = {
        "run": run_info(source=str(log), n_decisions=len(decisions)),
        "rungs": rungs,
        "fit": pick["fit"],
        "human_cost": run["human_cost"],
        "min_decisions": run["min_decisions"],
        "held_out": run["held_out"],
        "thresholds": thresholds,
        "prediction": {k: v for k, v in pick.items() if k not in ("rungs", "fit", "latency_ms")},
    }
    (log / "frozen-ladder.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(f"{' > '.join(rungs)} > human, {pick['fit']}, human_cost {run['human_cost']}, fitted on all {len(decisions)} test03 decisions")
    print(json.dumps(thresholds, indent=1))
    print(f"written {log / 'frozen-ladder.json'}")


MIN_COVERAGE = 0.95  # as benchmark.py score_all: a rung that answered less than this is not the ladder that was frozen


def short_rungs(rungs: list[str], arms: dict, labelled: set[int], share: float = MIN_COVERAGE) -> list[tuple[str, int]]:
    """The rungs that answered under `share` of the labelled decisions, with how many they answered. compose_ladder
    passes an unanswered decision to the next rung, so scoring a half-replayed rung would report a ladder never run."""
    return [(r, n) for r in rungs if (n := len(labelled & set(arms[r]))) < share * len(labelled)]


def confirm(log: Path, frozen_path: Path) -> None:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    rungs = frozen["rungs"]
    missing = [r for r in rungs if r not in arms]
    if missing:
        raise SystemExit(f"{log} has no answers from {missing}: replay the frozen ladder's arms first")
    labelled = {int(d["index_global"]) for d in decisions if d.get("truth") is not None}
    for r in rungs:
        print(f"{r}: answered {len(labelled & set(arms[r]))} of {len(labelled)} labelled decisions")
    short = short_rungs(rungs, arms, labelled)
    if short:
        raise SystemExit(f"{log}: {short} answered under {MIN_COVERAGE:.0%} of {len(labelled)} labelled decisions: rerun the replay (it resumes), then confirm")
    rows = compose_ladder(decisions, [(r, arms[r], frozen["thresholds"][r]) for r in rungs])
    rep = score(" > ".join(rungs), rows)
    iv = rep.interval or {}
    pred = frozen["prediction"]

    def block(prefix: str, rates: dict) -> dict:
        p = f"{prefix}_" if prefix else ""
        acc_lo, acc_hi = iv.get(f"{p}accuracy", (float("nan"), float("nan")))
        return {
            "n": rates.get("n", rep.n),
            "acted_wrong": [round(1 - rates["accuracy"], 4), [round(1 - acc_hi, 4), round(1 - acc_lo, 4)], pred.get(f"{p}acted_wrong")],
            "confidently_wrong": [round(rates["confidently_wrong"], 4), [round(x, 4) for x in iv.get(f"{p}confidently_wrong", ())], pred.get(f"{p}confidently_wrong")],
            "handed_to_person": [round(rates["human_rate"], 4), [round(x, 4) for x in iv.get(f"{p}human_rate", ())], pred.get(f"{p}human_rate")],
            "alone_right": [round(rates["alone_right"], 4), [round(x, 4) for x in iv.get(f"{p}alone_right", ())], pred.get(f"{p}alone_right")],
        }

    overall = block("", {"accuracy": rep.accuracy, "confidently_wrong": rep.confidently_wrong, "human_rate": rep.human_rate, "alone_right": rep.alone_right})
    judgement = block("judgement", rep.judgement or {})
    report = {
        "run": run_info(frozen=str(frozen_path), n_decisions=len(decisions)),
        "ladder": rungs,
        "fit": frozen["fit"],
        "thresholds": frozen["thresholds"],
        "format": "each rate: [test04 value, test04 95% by-episode interval, test03 ten-split prediction]",
        "overall": overall,
        "judgement": judgement,
        "by_rung": {name: sum(1 for r in rows if r["rung"] == name) for name in [*rungs, "human"]},
        "by_question": rep.by_question,
        "ece": round(rep.ece, 4),
    }
    for label, b in (("overall", overall), (f"judgement (all but {RULE_DECIDED})", judgement)):
        print(f"\n{label}, n {b['n']}:          test04 [95%]            test03 prediction")
        for k in ("acted_wrong", "confidently_wrong", "handed_to_person", "alone_right"):
            v, ci, p = b[k]
            print(f"  {k:18s} {v:.3f} [{ci[0]:.3f} {ci[1]:.3f}]   {p if p is None else f'{p:.3f}'}")
    print(f"\nanswered by: {report['by_rung']}")
    (log / "confirmation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {log / 'confirmation.json'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("freeze", "confirm"))
    p.add_argument("--log", required=True)
    p.add_argument("--frozen", default=None, help="confirm: the frozen-ladder.json written by freeze")
    a = p.parse_args()
    if a.command == "freeze":
        freeze(Path(a.log))
    else:
        if not a.frozen:
            raise SystemExit("confirm needs --frozen")
        confirm(Path(a.log), Path(a.frozen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
