"""Cell sensors: a load cell under each tray, and the rule that turns its readings into an in_bin answer.

Why a sensor and not the camera: on test03 and test04 the camera model called every wrong-bin landing "in the bin",
at 0.9 confidence or more (VALIDATION.md, "Where the frozen ladder hands off"). A tray scale cannot be fooled that
way: the weight lands on one tray or the other. Professional cells confirm a physical state with a physical sensor
and keep vision for what sensors cannot see.

The scale is simulated: the true force comes from the simulator (``Simulator.tray_load``), and the noise model below
stands in for the device. Nothing here is a real device; the numbers are chosen and written down before any run.

Noise model (written Oct 20, before any log uses it):
- a 5 kg straight-bar load cell under each tray (the TAL220B class sold by SparkFun), listed error +-0.05% of full
  scale = +-2.5 g (from SparkFun's forum on the listing; the maker's datasheet was not read). Modelled as Gaussian
  reading noise with sigma 2.5 g, pessimistic for an error band.
- a 1 g reading step (chosen, not from a datasheet).
- each reading is the mean of the samples over ``window_s`` (0.2 s, two samples of a 10 Hz HX711-class converter;
  here every physics step is a sample, then the noise is added once to the mean).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .decider import Question, RungAnswer, uniform
from .scene import Layout, tray_box_names
from .sim.base import Simulator

G = 9.81


@dataclass(frozen=True)
class LoadCellSpec:
    sigma_g: float = 2.5  # Gaussian reading noise, grams
    step_g: float = 1.0  # reading resolution, grams
    window_s: float = 0.2  # each reading averages the true force over this long


@dataclass
class TrayScales:
    """One load cell per tray. ``read`` returns grams per tray: true force averaged over the window, plus noise."""

    layout: Layout
    spec: LoadCellSpec = field(default_factory=LoadCellSpec)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng([self.seed, 2718])

    def true_grams(self, sim: Simulator) -> dict[str, float]:
        """Average true load per tray over the window, in grams (steps the sim: the scale integrates over time)."""
        n = max(1, round(self.spec.window_s / sim.dt))
        sums = {b: 0.0 for b in self.layout.bin_names}
        for _ in range(n):
            sim.step(1)
            for b in self.layout.bin_names:
                sums[b] += sim.tray_load(tray_box_names(b))
        return {b: sums[b] / n / G * 1000.0 for b in self.layout.bin_names}

    def noisy(self, grams: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for b, g in grams.items():
            reading = g + self._rng.normal(0.0, self.spec.sigma_g)
            out[b] = round(reading / self.spec.step_g) * self.spec.step_g if self.spec.step_g > 0 else reading
        return out

    def read(self, sim: Simulator) -> tuple[dict[str, float], dict[str, float]]:
        """(reading, true) in grams per tray."""
        true = self.true_grams(sim)
        return self.noisy(true), true

    @property
    def part_grams(self) -> float:
        return self.layout.part_rho * float(np.prod(self.layout.part_size)) * 1000.0


# In part weights: a tray that gained under QUIET took nothing; one that gained between LOW and HIGH took the part.
# Anything else (a part leaning on a rim, two parts' worth, a jolt) is unclear and goes up the ladder. With sigma 2.5 g
# on tare and reading, each margin (a quarter part, 12.8 g) is about 3.6 sigma of their difference.
QUIET, LOW, HIGH = 0.25, 0.75, 1.25


def scale_verdict(tare: Mapping[str, float], after: Mapping[str, float], intended: str, part_grams: float) -> tuple[str | None, str]:
    """The in_bin answer from the tray scales, or None when the readings do not settle it (then the camera, then a person).

    Returns (answer, where): where is "intended", "other bin", "no bin" or "unclear".
    """
    gain = {b: after[b] - tare.get(b, 0.0) for b in after}
    landed = {b for b, g in gain.items() if LOW * part_grams <= g <= HIGH * part_grams}
    quiet = {b for b, g in gain.items() if abs(g) < QUIET * part_grams}
    if intended not in gain:
        return None, "unclear"
    others = [b for b in gain if b != intended]
    if intended in landed and all(b in quiet for b in others):
        return "yes", "intended"
    if intended in quiet and len(landed) == 1 and all(b in quiet or b in landed for b in others):
        return "no", "other bin"
    if len(quiet) == len(gain):
        return "no", "no bin"
    return None, "unclear"  # both trays moved, a partial load (a part leaning on a rim), or more than one part's worth


class ScaleRung:
    """A rung that answers in_bin from the tray scales: certain when the rule settles it, no opinion otherwise.

    It sits first on the in_bin ladder with threshold 1.0, so only a settled reading is accepted; an unclear one leaves
    probability 0.5 each and the ladder climbs to the camera model, then the person.
    """

    def __init__(self, reading: Callable[[], Mapping[str, Any] | None], name: str = "scale") -> None:
        self.reading = reading
        self.name = name

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer:
        r = self.reading()
        if question.id != "in_bin" or r is None:
            return RungAnswer(self.name, uniform(question.options), 0.0, "no scale reading for this question")
        verdict = r.get("verdict")
        extra = {"where": r.get("where")}
        if verdict not in question.options:
            return RungAnswer(self.name, uniform(question.options), 0.0, None, extra)
        return RungAnswer(self.name, {o: (1.0 if o == verdict else 0.0) for o in question.options}, 0.0, None, extra)


# ----- the load cells as a person watches them: the panel's sensor window and the run log -----


def tray_grams(sim: Simulator, bins: tuple[str, ...]) -> dict[str, float]:
    """The true load on each tray right now, in grams: one contact-force read per tray, no stepping, so it is cheap
    enough for a step hook (``TrayScales.read`` steps the sim for its 0.2 s window and must never run in one)."""
    return {b: sim.tray_load(tray_box_names(b)) / G * 1000.0 for b in bins}


def parts_on(grams: float, part_grams: float) -> tuple[int, bool]:
    """(whole parts, settled): settled when the load is within QUIET of a whole number of parts."""
    n = grams / part_grams
    k = max(0, round(n))
    return k, abs(n - k) < QUIET


def part_counts(grams: Mapping[str, float], part_grams: float) -> dict[str, int]:
    return {b: parts_on(g, part_grams)[0] for b, g in grams.items()}


def load_cell_rows(grams: Mapping[str, float], part_grams: float, answering: bool) -> list[dict[str, str]]:
    """The panel window: one row per tray (grams and parts), then whether the scales answer in_bin or only show."""
    rows = []
    for b, g in grams.items():
        k, settled = parts_on(g, part_grams)
        detail = f"{k} part{'' if k == 1 else 's'}" if settled else f"{g / part_grams:.2f} parts: not a whole part (leaning on a rim, the hand touching, or still moving)"
        rows.append({"label": f"{b} tray", "value": f"{g:.1f} g", "detail": detail, "state": "ok" if settled else "warn"})
    if answering:
        rows.append({"label": "in_bin", "value": "scales answer", "detail": "--scales: a settled reading answers in_bin before the models", "state": "ok"})
    else:
        rows.append({"label": "in_bin", "value": "display only", "detail": "no --scales: the camera model and you answer in_bin", "state": "idle"})
    return rows


def load_cell_line(grams: Mapping[str, float], part_grams: float, before: Mapping[str, int] | None = None, label: str = "load cells") -> str:
    """One log line: trays whose part count changed first, with the change ("rework 51.0 g (+1 part), good 0.2 g")."""
    now = part_counts(grams, part_grams)
    changed = [b for b in grams if before is not None and now[b] != before.get(b, 0)]
    shown = []
    for b in [*changed, *(b for b in grams if b not in changed)]:
        text = f"{b} {grams[b]:.1f} g"
        if b in changed:
            delta = now[b] - before.get(b, 0)  # type: ignore[union-attr]
            text += f" ({delta:+d} part{'' if abs(delta) == 1 else 's'})"
        shown.append(text)
    return f"{label}: " + ", ".join(shown)


@dataclass
class LoadCellWatch:
    """Turns a stream of tray readings into the panel rows and a log line each time a tray's part count changes.

    A tray's count is logged only once it has held, settled, for ``settle_reads`` readings in a row: a part landing
    bounces, and the contact force spikes while it does. Each tray settles on its own, so a part left leaning on one
    tray's rim (a reading that never settles) does not hide a part landing in the other tray. The first line waits
    until every tray has held once.
    """

    part_grams: float
    answering: bool = False
    settle_reads: int = 3
    extra_rows: list[dict[str, str]] = field(default_factory=list)  # shown under the trays, e.g. the last in_bin check
    _logged: dict[str, int] | None = field(default=None, init=False)
    _candidate: dict[str, int] = field(default_factory=dict, init=False)
    _held: dict[str, int] = field(default_factory=dict, init=False)

    def update(self, grams: Mapping[str, float]) -> tuple[list[dict[str, str]], str | None]:
        rows = load_cell_rows(grams, self.part_grams, self.answering) + list(self.extra_rows)
        confirmed: dict[str, int] = {}  # trays whose count has held, settled, for settle_reads readings
        for b, g in grams.items():
            k, settled = parts_on(g, self.part_grams)
            if not settled:
                self._candidate.pop(b, None)
                self._held[b] = 0
                continue
            self._held[b] = self._held.get(b, 0) + 1 if self._candidate.get(b) == k else 1
            self._candidate[b] = k
            if self._held[b] >= self.settle_reads:
                confirmed[b] = k
        if self._logged is None:
            if len(confirmed) < len(grams):
                return rows, None
            self._logged = confirmed
            return rows, load_cell_line(grams, self.part_grams, label=f"load cells ({'answering in_bin' if self.answering else 'display only'})")
        changed = {b: k for b, k in confirmed.items() if self._logged.get(b) != k}
        if not changed:
            return rows, None
        # the change is shown only on the trays that settled at a new count; a tray still moving shows its grams alone
        now = part_counts(grams, self.part_grams)
        before = {b: self._logged.get(b, 0) if b in changed else now[b] for b in grams}
        self._logged.update(changed)
        return rows, load_cell_line(grams, self.part_grams, before)


def scale_check_line(part: str, scale: Mapping[str, Any]) -> str:
    """One log line per in_bin check the scales answered: tare, reading, gain in parts, verdict and where."""
    part_g = float(scale["part_g"])
    tare, now = scale["tare_g"], scale["reading_g"]
    gain = ", ".join(f"{b} {(now[b] - tare.get(b, 0.0)) / part_g:+.2f}" for b in now)
    verdict = scale["verdict"] or "unclear, up the ladder"
    true = scale.get("true_g") or {}
    truth = f"; sim truth {', '.join(f'{b} {g:.1f} g' for b, g in true.items())}" if true else ""
    return (
        f"scales: {part} for the {scale['intended']} tray: tare {', '.join(f'{b} {g:.0f} g' for b, g in tare.items())}; "
        f"now {', '.join(f'{b} {g:.0f} g' for b, g in now.items())}; gain in parts {gain} => {verdict} ({scale['where']}){truth}"
    )


def scale_check_row(part: str, scale: Mapping[str, Any]) -> dict[str, str]:
    """The last in_bin check as a row under the trays in the panel window."""
    verdict = scale["verdict"]
    return {"label": "last in_bin", "value": f"{part}: {verdict or 'unclear'}", "detail": f"{scale['where']} (meant for {scale['intended']})", "state": "ok" if verdict == "yes" else "warn"}


def sensor_window_hook(sim: Simulator, status: Any, every: int = 10, skip: tuple[str, ...] = ()) -> Callable[..., None]:
    """A step hook that puts every sensor the robot carries on the panel (``Simulator.sensor_windows``: the IMUs, the
    arm's joint torques, the tactile skin, the fingertips) every ``every`` steps, as ``status.set_sensor`` windows.
    Reads only, never steps. ``skip`` leaves out windows whose name contains any of those words (one another window
    already shows). A failing read turns the display off with one log line; it never stops the run."""
    state = {"steps": 0, "broken": False}

    def hook(*_args: Any) -> None:
        state["steps"] += 1
        if state["steps"] % every or state["broken"]:
            return
        try:
            for name, rows in sim.sensor_windows().items():
                if not any(word in name for word in skip):
                    status.set_sensor(name, rows)
        except Exception as error:  # noqa: BLE001  (a display must never stop the run)
            state["broken"] = True
            status.log(f"sensor windows: display off, a sensor read failed ({type(error).__name__}: {error})")

    return hook
