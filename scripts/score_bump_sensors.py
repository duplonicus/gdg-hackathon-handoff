"""Score every bump sensor on the probe's recorded crossings: pick each one's threshold on some seeds, score on others.

Reads what ``scripts/probe_bump_sensors.py`` wrote (``trials.jsonl`` plus one ``.npz`` per crossing) and runs the SAME
streaming detectors the robot runs online (``handoff.bump``) over each crossing's per-step readings, at the detector
settings of ``config/bump.yaml`` (baseline, observer gain, tares, hold; ``--config`` for another file). Only the
threshold is picked here:

- **Pick** (``--pick-seeds``): on the clean crossings (no true contact at all), the lowest threshold that keeps the
  false alarms within ``--budget`` per clean move (default 0.01: at most 1 in 100; with fewer than 100 clean moves
  that means none), times ``--margin``.
- **Score** (``--score-seeds``, never the pick seeds): bumps caught and missed, alarms before the true contact
  ("early", a false alarm on a bump crossing), false alarms per clean move, and the detection latency in ms from the
  first true contact (the contact solver, ``--truth any``, the default: the hand or the part it carries, since the probe cuts each crossing at either; ``hand``: the hand's links only)
  to the step the detector fires.

``contact_truth`` is scored too, as the reference: it is the truth, scored here on truth from every step, so its latency
comes out 0 (online it is checked every step of a guarded move too).

    .\\run.ps1 python scripts/score_bump_sensors.py output/probe/bump_sensors/<timestamp> --pick-seeds 1-10 --score-seeds 11-20

Prints the table and writes ``<probe dir>/score.json`` (or ``--out``). The pick and score seeds must not overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.bump import BUMP_SENSORS, DEFAULT_BUMP, IMU_SENSORS, BumpConfig, first_alarm, held, make_detector, placement, statistics  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    """'1-10', '1,3,5' or '1-3,7' -> the seeds."""
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def load_trials(probe_dir: Path) -> list[dict[str, Any]]:
    """Every crossing the probe recorded: its trials.jsonl line, with its arrays under "arrays"."""
    trials = []
    for line in (probe_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        if meta.get("skipped"):
            continue
        with np.load(probe_dir / meta["file"], allow_pickle=False) as data:
            meta["arrays"] = {k: data[k] for k in data.files}
        trials.append(meta)
    return trials


def truth_series(arrays: dict[str, np.ndarray], truth: str) -> np.ndarray:
    """Per step, whether the contact solver says the hand (or with ``any`` also the carried part) touches a guarded box."""
    hand = np.asarray(arrays["truth_hand"]) > 0
    return hand | (np.asarray(arrays["truth_part"]) > 0) if truth == "any" else hand


def samples(trial: dict[str, Any], sensor: str, truth: str) -> list[tuple[Any, ...]]:
    """One detector input per step, from a crossing's recorded readings, as the online BumpGuard feeds it."""
    a = trial["arrays"]
    if sensor in IMU_SENSORS:
        where = list(trial["placements"]).index(placement(sensor, trial.get("arm", "left")))
        return [(acc,) for acc in a["imu_acc"][:, where]]
    if sensor == "joint_torque":
        return [((tau, dp, bias) if np.all(np.isfinite(dp)) and np.all(np.isfinite(bias)) else (None,)) for tau, dp, bias in zip(a["torque"], a["dp"], a["bias"], strict=True)]
    if sensor == "tactile":
        return [(f,) for f in a["tactile"]]
    if sensor == "contact_truth":
        return [(float(c),) for c in truth_series(a, truth)]
    raise ValueError(sensor)


def available(trials: list[dict[str, Any]]) -> list[str]:
    """The bump sensors every crossing recorded a reading for."""
    out = []
    for sensor in BUMP_SENSORS:
        ok = True
        for t in trials:
            a = t["arrays"]
            if sensor in IMU_SENSORS:
                ok &= placement(sensor, t.get("arm", "left")) in list(t["placements"]) and a["imu_acc"].size > 0
            elif sensor == "joint_torque":
                ok &= a["torque"].size > 0
            elif sensor == "tactile":
                ok &= a["tactile"].size > 0
        if ok:
            out.append(sensor)
    return out


def trace_stats(trial: dict[str, Any], sensor: str, cfg: BumpConfig, truth: str) -> tuple[np.ndarray, int]:
    """(the detector's statistic per step, its hold) on one crossing."""
    det = make_detector(sensor, cfg, float(trial["dt"]))
    return statistics(det, samples(trial, sensor, truth)), det.hold


def first_truth(trial: dict[str, Any], truth: str) -> int | None:
    hits = np.nonzero(truth_series(trial["arrays"], truth))[0]
    return int(hits[0]) if len(hits) else None


def pick_threshold(clean_peaks: list[float], budget: float, margin: float) -> float:
    """The lowest threshold with at most ``floor(budget * n)`` clean moves over it (a move is over when its held peak
    exceeds the threshold strictly), times ``margin``."""
    if not clean_peaks:
        raise SystemExit("no clean crossing among the pick seeds: cannot pick a threshold")
    allowed = int(math.floor(budget * len(clean_peaks) + 1e-9))
    peaks = sorted(clean_peaks, reverse=True)
    return float(peaks[min(allowed, len(peaks) - 1)]) * margin if allowed < len(peaks) else 0.0


def score(trials: list[dict[str, Any]], cfg: BumpConfig, pick: list[int], score_seeds: list[int], *, budget: float = 0.01, margin: float = 1.0, truth: str = "hand") -> dict[str, Any]:
    overlap = sorted(set(pick) & set(score_seeds))
    if overlap:
        raise SystemExit(f"pick and score seeds overlap: {overlap}")
    out: dict[str, Any] = {"pick_seeds": pick, "score_seeds": score_seeds, "budget_per_clean_move": budget, "margin": margin, "truth": truth, "sensors": {}}
    picked = [t for t in trials if t["seed"] in pick]
    scored = [t for t in trials if t["seed"] in score_seeds]
    if not scored:
        raise SystemExit("no crossing among the score seeds")
    for sensor in available(trials):
        cache = {id(t): trace_stats(t, sensor, cfg, truth) for t in picked + scored}
        if sensor == "contact_truth":
            threshold = 0.0
        else:
            peaks = [float(np.max(held(*cache[id(t)]))) if len(cache[id(t)][0]) else 0.0 for t in picked if first_truth(t, truth) is None]
            threshold = pick_threshold(peaks, budget, margin)
        caught = missed = early = false_alarms = clean = 0
        latencies: list[float] = []
        for t in scored:
            stats, hold = cache[id(t)]
            alarm = first_alarm(stats, threshold, hold)
            contact = first_truth(t, truth)
            if contact is None:
                clean += 1
                false_alarms += alarm is not None
            elif alarm is None:
                missed += 1
            elif alarm < contact:
                early += 1
            else:
                caught += 1
                latencies.append((alarm - contact) * float(t["dt"]) * 1000.0)
        bumps = caught + missed + early
        lat = np.array(latencies)
        out["sensors"][sensor] = {
            "threshold": threshold, "unit": make_detector(sensor, cfg, 0.01).unit, "bumps": bumps, "caught": caught,
            "missed": missed, "early": early, "clean_moves": clean, "false_alarms": false_alarms,
            "false_alarms_per_clean_move": (false_alarms / clean) if clean else None,
            "latency_ms_median": float(np.median(lat)) if len(lat) else None,
            "latency_ms_p90": float(np.percentile(lat, 90)) if len(lat) else None,
            "latency_ms_max": float(np.max(lat)) if len(lat) else None,
            "truth_only": sensor == "contact_truth",
        }
    robot = {k: v for k, v in out["sensors"].items() if not v["truth_only"] and v["bumps"] and (v["false_alarms_per_clean_move"] or 0.0) <= budget}
    out["winner"] = (
        min(robot, key=lambda k: (-robot[k]["caught"] / robot[k]["bumps"], robot[k]["latency_ms_median"] if robot[k]["latency_ms_median"] is not None else math.inf))
        if robot
        else None
    )
    out["winner_rule"] = "most bumps caught on the score seeds, then the lowest median latency, among the robot's sensors within the false-alarm budget there"
    return out


def table(result: dict[str, Any]) -> str:
    head = f"{'sensor':14s} {'threshold':>10s} {'unit':8s} {'caught':>7s} {'missed':>7s} {'early':>6s} {'FA/clean':>9s} {'lat med':>8s} {'lat p90':>8s} {'lat max':>8s}"
    lines = [head, "-" * len(head)]

    def ms(v: float | None) -> str:
        return "-" if v is None else f"{v:.0f}"

    for name, s in result["sensors"].items():
        fa = "-" if s["false_alarms_per_clean_move"] is None else f"{s['false_alarms_per_clean_move']:.3f}"
        label = name + (" (truth)" if s["truth_only"] else "")
        lines.append(
            f"{label:14s} {s['threshold']:10.3f} {s['unit']:8s} {s['caught']:>3d}/{s['bumps']:<3d} {s['missed']:>7d} {s['early']:>6d} {fa:>9s} "
            f"{ms(s['latency_ms_median']):>8s} {ms(s['latency_ms_p90']):>8s} {ms(s['latency_ms_max']):>8s}"
        )
    lines.append(f"winner: {result['winner']} ({result['winner_rule']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--pick-seeds", required=True, help="seeds to pick thresholds on, e.g. 1-10 (quote a comma list in PowerShell)")
    parser.add_argument("--score-seeds", required=True, help="seeds to score on, e.g. 11-20 (no overlap with --pick-seeds)")
    parser.add_argument("--budget", type=float, default=0.01, help="false alarms allowed per clean move when picking (default 0.01)")
    parser.add_argument("--margin", type=float, default=1.0, help="multiply each picked threshold by this (default 1.0)")
    parser.add_argument("--truth", choices=("hand", "any"), default="any", help="what counts as a true bump: the hand or the part it carries touching (any, the default: the probe cuts each crossing at either), or the hand's links only (hand)")
    parser.add_argument("--config", type=Path, default=DEFAULT_BUMP, help="detector settings (default config/bump.yaml; its thresholds are ignored)")
    parser.add_argument("--out", type=Path, default=None, help="default: <probe_dir>/score.json")
    args = parser.parse_args(argv)
    cfg = BumpConfig.from_yaml(args.config)
    result = score(load_trials(args.probe_dir), cfg, parse_seeds(args.pick_seeds), parse_seeds(args.score_seeds), budget=args.budget, margin=args.margin, truth=args.truth)
    result["probe_dir"] = str(args.probe_dir)
    result["config"] = str(args.config)
    print(table(result))
    out = args.out or args.probe_dir / "score.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
