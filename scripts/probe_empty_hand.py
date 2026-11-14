"""Genesis CPU probe of the EMPTY left hand: where does it start pressing on itself, and what do the fingertips read?

Two experiments on one build, the part parked 5 m away so nothing is in the hand:

- **ramp**: close slowly at the grip's slow rate (GRIP_SLOW_RATE, 0.25 closure/s, a read every 2 steps) from 0.5 and
  record the measured closure, the command and the force at each tip's first reading; then reopen and run the force
  grasp (``grasp_control="force"``) on the empty hand through the real skill. Backs ``skills.EMPTY_CLOSURE``: on Oct 21
  (one CPU run) the index and thumb tips first touched each other at a measured closure of 0.818 (command 0.845) and
  read 5.61 and 5.80 N at once, and the empty force grasp failed with "the fingers pressed on the hand itself at
  closure 0.86".
- **snap**: command the hand fully closed at once and read the tips and every contact on a tip link every 10 steps
  for 120 steps. On Oct 21 the index and thumb tips saturated at 20 N (the FSR's range) from step 30, pressing on each
  other, and at step 60 the middle, ring and little tips read about 20 N against the palm (left_gripper_base).

Runs on gs.cpu with the decimated robot and the cameras off (no GPU), about a minute:

    .\\run.ps1 python scripts/probe_empty_hand.py
    .\\run.ps1 python scripts/probe_empty_hand.py --mode ramp
    .\\run.ps1 python scripts/probe_empty_hand.py --mode snap

Prints each mode's summary as one JSON line (also written to output/probe/empty_hand/<timestamp>/result.json). With
``--mode both`` (the default) the snap runs after a ``sim.reset()``, not on a fresh build.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_grip_cpu import cpu_config  # noqa: E402

from handoff import skills  # noqa: E402
from handoff.scene import Layout  # noqa: E402
from handoff.sim.base import BoxSpec, SceneSpec, fingertip_links  # noqa: E402

ARM = "left"


def ramp(sim) -> dict:
    first: dict[str, dict[str, float]] = {}
    cmd = 0.5
    sim.set_hand(ARM, cmd)
    sim.settle(0.5)
    while cmd < 1.0:
        cmd = min(1.0, cmd + skills.GRIP_SLOW_RATE * sim.dt * skills.GRIP_READ_EVERY)
        sim.set_hand(ARM, cmd)
        sim.step(skills.GRIP_READ_EVERY)
        tips = sim.read_fingertips(ARM)
        closure = sim.hand_closure(ARM)
        for k, v in tips.items():
            if v > 0 and k not in first:
                first[k] = {"closure": round(closure, 3), "cmd": round(cmd, 3), "n": round(v, 2),
                            "finger_side_n": round(skills.finger_side_n(tips, ARM), 2)}
    sim.set_hand(ARM, 0.0)
    sim.settle(1.0)
    reopened = {"closure": round(sim.hand_closure(ARM), 3), "tips": {k: round(v, 2) for k, v in sim.read_fingertips(ARM).items()}}
    ctx = skills.TaskContext(layout=Layout(), target_part="part", grasp_control="force")
    r = skills.SKILLS["grasp"](sim, ctx)
    return {"mode": "ramp", "rate_per_s": skills.GRIP_SLOW_RATE, "first_touch": first, "reopened": reopened,
            "empty_closure_const": skills.EMPTY_CLOSURE, "force_grasp_ok": r.ok, "force_grasp_reason": r.reason,
            "grip_line": skills.grip_line("part", r)}


def snap(sim, steps: int = 120, every: int = 10) -> dict:
    kit = sim.kit
    tip_idx = {int(kit.get_link(link).idx) for pair in fingertip_links(ARM).values() for link in pair}
    name = {int(kit.get_link(n).idx): n for n in kit.link_names}
    rows = []
    sim.set_hand(ARM, 1.0)
    for i in range(1, steps + 1):
        sim.step(1)
        if i % every:
            continue
        c = kit.robot.get_contacts()
        a = sim._np(c["link_a"]).astype(int).ravel()
        b = sim._np(c["link_b"]).astype(int).ravel()
        pairs = sorted({(name.get(x, str(x)), name.get(y, str(y))) for x, y in zip(a, b, strict=True) if x in tip_idx or y in tip_idx})
        tips = {k.removeprefix(f"{ARM}_"): round(v, 2) for k, v in sim.read_fingertips(ARM).items()}
        rows.append({"step": i, "closure": round(sim.hand_closure(ARM), 3), "tips": tips, "tip_contacts": [list(p) for p in pairs]})
        print(i, "closure", rows[-1]["closure"], tips, "tip contacts:", pairs[:6], flush=True)
    tip_max = sim.sensors.fingertip.max_force
    return {"mode": "snap", "fingertip_max_n": tip_max, "rows": rows,
            "first_saturated_step": {t: next((r["step"] for r in rows if r["tips"].get(t, 0.0) >= tip_max), None) for t in rows[0]["tips"]} if rows else {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("ramp", "snap", "both"), default="both")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / "output" / "probe" / "empty_hand" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    from handoff.sim.genesis_sim import GenesisSimulator

    t0 = time.time()
    sim = GenesisSimulator(str(cpu_config(out)))
    sim.build(SceneSpec(boxes=(BoxSpec("part", (0.04, 0.04, 0.08), (5.0, 5.0, 0.04)),)))
    print(f"built {time.time() - t0:.0f} s", flush=True)
    results = []
    if args.mode in ("ramp", "both"):
        results.append(ramp(sim))
        print(json.dumps(results[-1]), flush=True)
    if args.mode in ("snap", "both"):
        if results:
            sim.reset()
        results.append(snap(sim))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "rows"}), flush=True)
    summary = {"results": results, "wall_s": round(time.time() - t0, 1)}
    (out / "result.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
