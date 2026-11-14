"""Greedy versus joint threshold fits for the multi-rung ladders, over five held-out splits and across logs.

The joint fit is arXiv 2610.21412's per-choice rule (a lower rung answers only where it beats the next rung, not
the person); see docs/papers/README.md. Offline, no model calls:

    .\run.ps1 python scripts/compare_threshold_fits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B

from handoff.benchmark import (
    choose_thresholds,
    choose_thresholds_joint,
    choose_thresholds_joint_per_question,
    choose_thresholds_per_question,
    compose_ladder,
    score,
    split,
)

log = Path("output/sort/test03")
decisions = B.load(log); arms = B.load_arms(log); B.collapse_finish(decisions, arms)
x = Path("output/sort/log02b"); xd = B.load(x); xa = B.load_arms(x); B.collapse_finish(xd, xa)
ladders = {"A 2b>artemis": ["decider-2b-vision", "pplx-decider-artemis-vision"], "C 2b>gemma": ["decider-2b-vision", "gemma-e4b"]}
fits = {"greedy": choose_thresholds, "greedy/q": choose_thresholds_per_question, "joint": choose_thresholds_joint, "joint/q": choose_thresholds_joint_per_question}
for name, rn in ladders.items():
    for fname, fit in fits.items():
        vals = []
        for seed in range(5):
            held, test = split(decisions, 0.3, seed)
            t = fit(held, [(r, arms[r]) for r in rn])
            rep = score("x", compose_ladder(test, [(r, arms[r], t[r]) for r in rn]), bootstrap=10)
            vals.append((rep.alone_right, rep.human_rate, rep.confidently_wrong))
        t = fit(xd, [(r, xa[r]) for r in rn]) if all(r in xa for r in rn) else None
        if t:
            rep = score("x", compose_ladder(decisions, [(r, arms[r], t[r]) for r in rn]), bootstrap=10)
            print(f"{name:14s} {fname:9s} thresholds from log02b: alone {rep.alone_right:.3f} human {rep.human_rate:.3f} cwrong {rep.confidently_wrong:.3f}")
        m = [sum(v[i] for v in vals) / 5 for i in range(3)]
        print(f"{name:14s} {fname:9s} alone {m[0]:.3f} human {m[1]:.3f} cwrong {m[2]:.3f}   per seed cwrong {[round(v[2],3) for v in vals]}")
