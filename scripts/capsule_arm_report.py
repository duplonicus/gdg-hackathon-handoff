"""Every replayed arm on a capsule log, with 95% by-shift bootstrap intervals (docs/PREREG_CAPSULES.md, "What gets
reported", items 1 to 3; item 4, the frozen ladder, is freeze_ladder.py confirm).

On the leaker calls (leak_block, leak_capsule): leakers caught, leaker pictures confidently missed (answered good / no
at 0.9 or more), clean pictures called leakers. On all questions: wrong, and wrong at 0.9 or more. Stdlib only.

    python3 scripts/capsule_arm_report.py output/capsules/test05 [--resamples 2000] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

LEAKER_CALLS = ("leak_block", "leak_capsule")
POSITIVE = {"yes", "leaker"}
SURE = 0.9


def rows_for(log: Path) -> dict[str, list[dict]]:
    """Each arm's scored answers: question, shift, truth, top answer and its probability. Errored rows and
    unlabelled decisions are left out; a decision answered twice (an error, then its retry) counts once, last wins."""
    decisions = [json.loads(line) for line in (log / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in (log / "arms.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        i = int(r["index_global"])
        d = decisions[i]
        if d.get("truth") is None:
            continue
        top = max(r["probs"], key=r["probs"].get)
        by_arm[r["arm"]][i] = {"question": d["question"], "episode": d["episode"], "truth": d["truth"], "answer": top, "conf": r["probs"][top]}
    return {arm: list(rows.values()) for arm, rows in by_arm.items()}


def counts(rows: list[dict]) -> dict[str, int]:
    c = defaultdict(int)
    for r in rows:
        wrong = r["answer"] != r["truth"]
        c["n"] += 1
        c["wrong"] += wrong
        c["sure_wrong"] += wrong and r["conf"] >= SURE
        if r["question"] in LEAKER_CALLS:
            if r["truth"] in POSITIVE:
                c["leakers"] += 1
                c["caught"] += not wrong
                c["sure_missed"] += wrong and r["conf"] >= SURE
            else:
                c["clean"] += 1
                c["false_alarm"] += wrong
    return c


def rates(c: dict[str, int]) -> dict[str, float]:
    def share(a: str, b: str) -> float:
        return c[a] / c[b] if c[b] else float("nan")

    return {
        "caught": share("caught", "leakers"),
        "sure_missed": share("sure_missed", "leakers"),
        "false_alarms": share("false_alarm", "clean"),
        "wrong_all": share("wrong", "n"),
        "sure_wrong_all": share("sure_wrong", "n"),
    }


def report(rows: list[dict], resamples: int, seed: int = 0) -> dict:
    """Point rates plus 2.5 and 97.5 percentiles over shifts drawn with replacement."""
    by_shift: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_shift[r["episode"]].append(r)
    shifts = sorted(by_shift)
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(resamples):
        pick = [r for s in rng.choices(shifts, k=len(shifts)) for r in by_shift[s]]
        for k, v in rates(counts(pick)).items():
            if v == v:  # not nan
                draws[k].append(v)
    c = counts(rows)
    out = {"counts": dict(c), "shifts": len(shifts)}
    for k, v in rates(c).items():
        ds = sorted(draws[k])
        lo, hi = (ds[int(0.025 * (len(ds) - 1))], ds[int(0.975 * (len(ds) - 1))]) if ds else (float("nan"), float("nan"))
        out[k] = [round(v, 4), [round(lo, 4), round(hi, 4)]]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    log = Path(a.log)
    result = {arm: report(rows, a.resamples) for arm, rows in sorted(rows_for(log).items())}
    print(f"{log}: 95% by-shift bootstrap, {a.resamples} resamples; leaker calls = {', '.join(LEAKER_CALLS)}")
    for arm, r in result.items():
        c = r["counts"]
        def f(k):
            v, (lo, hi) = r[k]
            return f"{v:.3f} [{lo:.3f} {hi:.3f}]"
        print(f"\n{arm}  ({c.get('leakers', 0)} leaker pictures, {c.get('clean', 0)} clean, {c['n']} answers in all, {r['shifts']} shifts)")
        print(f"  leakers caught         {f('caught')}")
        print(f"  confidently missed     {f('sure_missed')}  ({c.get('sure_missed', 0)} pictures)")
        print(f"  false alarms           {f('false_alarms')}")
        print(f"  all questions wrong    {f('wrong_all')}")
        print(f"  all, wrong at >= {SURE}  {f('sure_wrong_all')}  ({c.get('sure_wrong', 0)} answers)")
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwritten {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
