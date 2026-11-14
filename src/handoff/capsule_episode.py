"""One capsule episode: which cell has work, which capsules leak, how big each pool is; and putting that in the scene.

The scene is built once (``make_two_cell_scene``); an episode only moves what is already there:
- the grid of fixed capsules stands in the cell with work (under the floor when neither cell has any), each nudged and
  turned a little, as capsules on a real tray never line up exactly;
- a leaker is one of those same fixed capsules with a pool of one size under it. Nothing else about it differs: when
  leakers were loose bodies they settled tilted and turned, a give-away a model could read instead of the pool (Nov 4
  smoke run). The robot's pick swaps in a loose body at that moment (tasks/capsules.py);
- each cell shows its "work" look (full cart, empty good bin) or its "done" look (empty trays, full good bin).

Deterministic for a seed. The mix follows dup's description (Nov 4): leakers are a few per tray, easy for a person to
spot; the pool size is the difficulty dial, from a fresh leak barely past the capsule's edge to a spread one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capsule_cell import MAX_LEAKERS, MAX_SPARES, OFFSTAGE, PARK_Z, POOL_SIZES, TwoCells, look_boxes, parked_pos

LEAKER_COUNTS = (0, 1, 2, 3, 4, 5)
LEAKER_WEIGHTS = (0.15, 0.30, 0.25, 0.15, 0.10, 0.05)
WORK_CELLS = ("a", "b", None)
WORK_WEIGHTS = (0.45, 0.45, 0.10)  # one in ten: both cells finished, nowhere to go
JITTER_XY = 0.0008  # m each way: off its grid spot
JITTER_YAW = 8.0  # degrees each way: at 8 a 24 mm capsule still clears its neighbours' 5 mm gaps


@dataclass(frozen=True)
class CapsuleEpisode:
    seed: int
    work_cell: str | None  # "a", "b", or None when both cells are finished
    leakers: tuple[int, ...]  # grid spots of the leakers, in leaker-slot order
    pool_sizes: tuple[int, ...]  # index into POOL_SIZES, one per leaker

    def leaker_at(self, spot: int) -> int | None:
        """The leaker slot standing at a grid spot, or None for a good capsule."""
        return self.leakers.index(spot) if spot in self.leakers else None


def make_capsule_episode(cells: TwoCells, seed: int) -> CapsuleEpisode:
    rng = np.random.default_rng([seed, 4242])
    work = WORK_CELLS[int(rng.choice(len(WORK_CELLS), p=WORK_WEIGHTS))]
    if work is None:
        return CapsuleEpisode(seed, None, (), ())
    n = int(rng.choice(LEAKER_COUNTS, p=LEAKER_WEIGHTS))
    assert n <= MAX_LEAKERS
    spots = tuple(int(i) for i in rng.choice(len(cells.layout.grid()), size=n, replace=False))
    sizes = tuple(int(s) for s in rng.integers(0, len(POOL_SIZES), size=n))
    return CapsuleEpisode(seed, work, spots, sizes)


def capsule_poses(cells: TwoCells, episode: CapsuleEpisode) -> list[tuple[float, float, float]]:
    """Every grid capsule's world x, y and heading this episode: its spot in the cell with work (cell a's when there is
    none), nudged and turned by the episode's own random stream."""
    rng = np.random.default_rng([episode.seed, 99])
    grid = cells.layout.grid()
    cell = cells.cell(episode.work_cell or "a")
    nudges = rng.uniform(-1.0, 1.0, size=(len(grid), 3)) * np.array([JITTER_XY, JITTER_XY, JITTER_YAW])
    out = []
    for (gx, gy), (dx, dy, dyaw) in zip(grid, nudges, strict=True):
        x, y = cell.to_world(gx + dx, gy + dy)
        out.append((x, y, cell.yaw_deg + float(dyaw)))
    return out


def place_capsule_episode(sim, cells: TwoCells, episode: CapsuleEpisode) -> None:
    """Move everything into this episode's places. Safe to call between episodes in any order."""
    lay = cells.layout
    grid = lay.grid()
    cell = cells.cell(episode.work_cell) if episode.work_cell else None
    for name in (*cells.leaker_names, *cells.spare_names):
        sim.detach(name)  # nothing stays welded to a hand across episodes
    z = lay.paper_top + lay.capsule[2] / 2.0
    for i, (x, y, yaw) in enumerate(capsule_poses(cells, episode)):
        sim.set_object_pose(cells.grid_names[i], (x, y, z) if cell is not None else parked_pos(x, y, 0.0), yaw)
    poses = capsule_poses(cells, episode)
    for k in range(MAX_LEAKERS):
        for size in range(len(POOL_SIZES)):
            sim.set_object_pose(cells.pool_name(k, size), (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1] + 0.2 + 0.1 * size, PARK_Z), 0.0)
        if cell is not None and k < len(episode.leakers):
            x, y, _ = poses[episode.leakers[k]]
            sim.set_object_pose(cells.pool_name(k, episode.pool_sizes[k]), (x, y, lay.paper_top + 0.0003), 0.0)
        sim.set_object_pose(cells.leaker_names[k], (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1], lay.capsule[2] / 2.0), 0.0)
    for k, name in enumerate(cells.spare_names[:MAX_SPARES]):
        sim.set_object_pose(name, (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1] - 0.4, lay.capsule[2] / 2.0), 0.0)
    for each in cells.cells:
        shown = "work" if episode.work_cell == each.name else "done"
        for look in ("work", "done"):
            for box in look_boxes(each, look):
                pos = box.pos if look == shown else parked_pos(*box.pos)
                sim.set_object_pose(box.name, pos, box.yaw_deg)
