"""Film frames from in-sim cameras at a fixed rate of SIM time, for the video (media/film/SCRIPT.md, "Footage").

A recorder wraps ``sim.step``: after each step it saves one frame per camera for every 1/fps of sim time that has
passed, so a clip plays smooth whatever the sim's speed. ``frames.jsonl`` maps every frame to its sim time and wall
time (``time.time()``), which is how the operator page's screenshots, taken on the wall clock, are laid beside it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Phase 1 (the sort cell, handoff.scene.Layout): world metres. The robot drives from x -1.2 to x 0 facing +x; the
# table spans x 0.35 to 1.05 and y -0.40 to 0.80 at 0.82 m; the pick slots sit at y 0.15, the bins near (0.47, 0.47).
# name -> (position, look-at, fov degrees)
SORT_ANGLES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], float]] = {
    "front": ((1.45, -0.35, 1.45), (0.40, 0.25, 0.95), 50),  # beyond the table: the robot's front, the slots, both trays
    "wide": ((2.60, -2.00, 2.02), (-0.30, 0.20, 1.02), 50),  # the drive in and the whole cell
    "trays": ((1.25, 1.20, 1.50), (0.35, 0.30, 1.00), 45),  # from the trays' side: landings, with the robot behind
}


def camera_specs(angles: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], float]], names: list[str], res: tuple[int, int]):
    """CameraSpecs named ``film_<angle>`` for the scene."""
    from .sim.base import CameraSpec

    unknown = [n for n in names if n not in angles]
    if unknown:
        raise ValueError(f"unknown film angles {unknown}; known: {', '.join(angles)}")
    return tuple(CameraSpec(f"film_{n}", pos=angles[n][0], lookat=angles[n][1], fov=angles[n][2], res=res) for n in names)


class FilmRecorder:
    """Saves <out>/<angle>/fNNNNN.jpg for each camera at ``fps`` of sim time, and <out>/frames.jsonl."""

    def __init__(self, sim: Any, out: Path, names: list[str], fps: float = 30.0, quality: int = 95) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.sim, self.out, self.names, self.fps, self.quality = sim, Path(out), list(names), fps, quality
        self.frames = 0
        self._next_t: float | None = None
        self._real_step = None
        self._index = None

    @staticmethod
    def holds_take(out: Path, names: list[str]) -> bool:
        """True when <out> already has a frame index or any frame for these angles."""
        return (Path(out) / "frames.jsonl").exists() or any(next((Path(out) / n).glob("f*.jpg"), None) is not None for n in names)

    def start(self) -> None:
        if self.holds_take(self.out, self.names):
            raise FileExistsError(f"{self.out} already holds a take: film into a new folder (frame numbers restart at 0)")
        for n in self.names:
            (self.out / n).mkdir(parents=True, exist_ok=True)
        self._index = (self.out / "frames.jsonl").open("w", encoding="utf-8")
        self._next_t = self.sim.time
        self._real_step = self.sim.step

        def step(n: int = 1) -> None:
            self._real_step(n)
            self.catch_up()

        self.sim.step = step

    def catch_up(self) -> None:
        """One frame per 1/fps of sim time passed; a long step can owe more than one (they show the same moment)."""
        from PIL import Image

        while self._next_t is not None and self.sim.time >= self._next_t:
            name = f"f{self.frames:05d}.jpg"
            for n in self.names:
                Image.fromarray(self.sim.capture(f"film_{n}").rgb).save(self.out / n / name, quality=self.quality)
            self._index.write(json.dumps({"frame": self.frames, "sim_t": round(self.sim.time, 4), "wall_t": round(time.time(), 3)}) + "\n")
            self.frames += 1
            self._next_t += 1.0 / self.fps

    def stop(self) -> None:
        self._next_t = None
        if self._real_step is not None:
            self.sim.step = self._real_step
            self._real_step = None
        if self._index is not None:
            self._index.close()
            self._index = None
