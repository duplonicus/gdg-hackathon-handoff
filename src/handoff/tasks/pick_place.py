"""Stage 1 task: pick one part off the table and place it in one bin, then go home.

A fixed sequence of skills. A step that fails ends the trial there, so the result says exactly
which skill broke; the final checks are computed from sim state either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..checks import any_on_floor, in_bin
from ..skills import SKILLS, TaskContext, final_state
from ..sim.base import Simulator

SEQUENCE = ("drive_to_table", "look_at_table", "reach", "approach", "grasp", "lift", "move_to_bin", "lower", "release", "go_home")
FRAME_EVENTS = {"approach": "pregrasp", "grasp": "closed", "lift": "lifted", "release": "released"}


@dataclass
class StepRecord:
    skill: str
    ok: bool
    reason: str
    t_start: float
    t_end: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    success: bool
    failed_at: str | None
    steps: list[StepRecord]
    checks: dict[str, Any]
    sim_time: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed_at": self.failed_at,
            "sim_time": self.sim_time,
            "checks": self.checks,
            "steps": [
                {"skill": s.skill, "ok": s.ok, "reason": s.reason, "t_start": s.t_start, "t_end": s.t_end, "data": _plain(s.data)}
                for s in self.steps
            ],
        }


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def run_pick_place(sim: Simulator, ctx: TaskContext) -> TrialResult:
    steps: list[StepRecord] = []
    failed_at: str | None = None
    for name in SEQUENCE:
        t0 = sim.time
        result = SKILLS[name](sim, ctx)
        steps.append(StepRecord(name, result.ok, result.reason, t0, sim.time, result.data))
        if name in FRAME_EVENTS:
            ctx.event(FRAME_EVENTS[name])
        if not result.ok:
            failed_at = name
            ctx.event(f"failed_{name}")
            break
    state = final_state(sim, ctx)  # let a dropped part land before judging where it ended up
    other_bins = [b for b in ctx.layout.bin_names if b != ctx.target_bin]
    checks = {
        "in_target_bin": in_bin(state, ctx.target_part, ctx.target_bin),
        "in_other_bin": any(in_bin(state, ctx.target_part, b) for b in other_bins),
        "on_floor": any_on_floor(state),
        "part_pos": state.objects[ctx.target_part].pos.tolist(),
    }
    success = failed_at is None and checks["in_target_bin"] and not checks["in_other_bin"] and not checks["on_floor"]
    return TrialResult(success, failed_at, steps, checks, sim.time)
