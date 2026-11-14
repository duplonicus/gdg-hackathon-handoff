"""The scene as plain numbers: a table, two bin trays, a pool of parts, and the cameras.

World frame: the robot parks at the origin facing +x. The table is in front of it, the parts
stand on the table in a row along x, and both bins sit to the left of the row, because an IK
sweep (Oct 11) showed the left arm reaches wrist targets with y between 0.15 and 0.55 to within
6 mm but cannot cross the body midline with the grasp orientation. Other measured facts behind
these numbers: wrist reach about 0.63 m from the shoulder at (0, +-0.15, 1.20), hands resting at
z 0.885, base body extending about 0.30 m ahead of its origin.

Parts: Genesis builds a scene once (kernel compile), so every finish the episodes will ever
show must exist from the start. The **pool** holds ``pool_per_kind`` parts of each finish with
defect severities spread from faint to obvious, parked on the floor far from the table. An
**episode** picks three of them (by seed), stands them in the row slots with jitter, and leaves
the rest parked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .finish import Finish, finish_pool, texture
from .sim.base import BoxSpec, CameraSpec, SceneSpec


@dataclass(frozen=True)
class BinRegion:
    """The inside of a bin tray, for the "is the part in the bin" check."""

    name: str
    cx: float
    cy: float
    half_inner: float
    floor_z: float
    wall_top_z: float


@dataclass(frozen=True)
class Layout:
    table_top: float = 0.82
    table_x: tuple[float, float] = (0.35, 1.05)
    table_y: tuple[float, float] = (-0.40, 0.80)
    part_size: tuple[float, float, float] = (0.04, 0.04, 0.08)  # 5 cm does not fit the MH3 hand's pinch (Oct 11)
    slot_xy: tuple[tuple[float, float], ...] = ((0.55, 0.15), (0.73, 0.15), (0.91, 0.15))
    pool_per_kind: int = 6
    park_xy: tuple[float, float] = (3.0, 1.5)  # where unused pool parts wait, on the floor
    bin_inner: float = 0.16
    bin_wall_h: float = 0.08
    bin_thick: float = 0.01
    part_rho: float = 400.0  # kg/m^3: a 4 x 4 x 8 cm part weighs 51 g, what the line's tray scales expect per part
    bin_xy: tuple[tuple[str, float, float], ...] = (("good", 0.50, 0.38), ("rework", 0.44, 0.57))
    robot_start: tuple[float, float, float] = (-1.2, 0.0, 0.0)  # x, y, yaw_deg before the drive
    park_x: float = 0.0  # where the base origin should end up after the drive
    jitter_xy_m: float = 0.02
    jitter_yaw_deg: float = 15.0
    # How far each bin is moved per episode (m, each axis). A real cell's bin gets nudged; the robot has to find it
    # (perception.locate_bins), not read it from this file. 0 keeps the bins where bin_xy says.
    bin_jitter_m: float = 0.0

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(f"part_{i:02d}" for i in range(3 * self.pool_per_kind))

    @property
    def bin_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.bin_xy)

    @property
    def part_center_z(self) -> float:
        return self.table_top + self.part_size[2] / 2.0

    def slot_pos(self, slot: int) -> tuple[float, float, float]:
        x, y = self.slot_xy[slot]
        return (x, y, self.part_center_z)

    def park_pos(self, index: int) -> tuple[float, float, float]:
        x0, y0 = self.park_xy
        return (x0 + 0.1 * index, y0, self.part_size[2] / 2.0)

    def bin_region(self, name: str) -> BinRegion:
        for bin_name, cx, cy in self.bin_xy:
            if bin_name == name:
                floor_z = self.table_top + self.bin_thick
                return BinRegion(name, cx, cy, self.bin_inner / 2.0, floor_z, floor_z + self.bin_wall_h)
        raise KeyError(name)

    def bin_regions(self) -> dict[str, BinRegion]:
        return {name: self.bin_region(name) for name in self.bin_names}


def table_box(layout: Layout) -> BoxSpec:
    (x0, x1), (y0, y1) = layout.table_x, layout.table_y
    return BoxSpec(
        name="table",
        size=(x1 - x0, y1 - y0, layout.table_top),
        pos=((x0 + x1) / 2.0, (y0 + y1) / 2.0, layout.table_top / 2.0),
        color=(0.55, 0.40, 0.25),
        fixed=True,
    )


BIN_COLORS = {"good": (0.2, 0.6, 0.2), "rework": (0.8, 0.5, 0.1)}  # green tray, orange tray


TRAY_PIECES = ("floor", "wall_xm", "wall_xp", "wall_ym", "wall_yp")


def tray_box_names(name: str) -> tuple[str, ...]:
    """The fixed boxes of one tray, everything a load cell under it carries."""
    return tuple(f"{name}_{piece}" for piece in TRAY_PIECES)


def bin_boxes(layout: Layout, name: str) -> list[BoxSpec]:
    """A tray: a floor plate and four thin walls, all fixed so the robot cannot knock it over."""
    region = layout.bin_region(name)
    inner, t, h = layout.bin_inner, layout.bin_thick, layout.bin_wall_h
    outer = inner + 2 * t
    z_floor = layout.table_top + t / 2.0
    z_wall = region.floor_z + h / 2.0
    cx, cy = region.cx, region.cy
    color = BIN_COLORS[name]
    return [
        BoxSpec(f"{name}_floor", (outer, outer, t), (cx, cy, z_floor), color=color, fixed=True),
        BoxSpec(f"{name}_wall_xm", (t, outer, h), (cx - (inner + t) / 2.0, cy, z_wall), color=color, fixed=True),
        BoxSpec(f"{name}_wall_xp", (t, outer, h), (cx + (inner + t) / 2.0, cy, z_wall), color=color, fixed=True),
        BoxSpec(f"{name}_wall_ym", (outer, t, h), (cx, cy - (inner + t) / 2.0, z_wall), color=color, fixed=True),
        BoxSpec(f"{name}_wall_yp", (outer, t, h), (cx, cy + (inner + t) / 2.0, z_wall), color=color, fixed=True),
    ]


def part_pool(layout: Layout, seed: int = 0) -> list[tuple[str, Finish]]:
    return list(zip(layout.part_names, finish_pool(layout.pool_per_kind, seed), strict=True))


def part_boxes(layout: Layout, seed: int = 0) -> list[BoxSpec]:
    boxes = []
    for i, (name, finish) in enumerate(part_pool(layout, seed)):
        boxes.append(
            BoxSpec(name, layout.part_size, layout.park_pos(i), rho=layout.part_rho, friction=1.5, texture=texture(finish, seed=seed * 1000 + i), finish=finish.label)
        )
    return boxes


def cameras(layout: Layout) -> tuple[CameraSpec, ...]:
    sx, sy, sz = layout.slot_pos(0)
    bx = float(np.mean([cx for _, cx, _ in layout.bin_xy]))
    by = float(np.mean([cy for _, _, cy in layout.bin_xy]))
    return (
        # Close-up of the pick spot from the front-left, for reading grasp frames.
        CameraSpec("debug", pos=(sx + 0.40, sy + 0.40, sz + 0.19), lookat=(sx, sy, sz), fov=35),
        # The inspection camera over the pick station: sees the top and two side faces of the part.
        CameraSpec("inspect", pos=(sx - 0.10, sy + 0.09, sz + 0.20), lookat=(sx, sy, sz), fov=28),
        # The output camera over both bins, for "did it land in the bin" and "is the sort complete".
        CameraSpec("bins", pos=(bx - 0.25, by, layout.table_top + 0.55), lookat=(bx, by, layout.table_top + 0.05), fov=45),
    )


def make_scene(layout: Layout | None = None, seed: int = 0) -> SceneSpec:
    """Everything built once: the table, the bins, the whole part pool (parked) and the cameras."""
    layout = layout or Layout()
    boxes = [table_box(layout)]
    for name in layout.bin_names:
        boxes.extend(bin_boxes(layout, name))
    boxes.extend(part_boxes(layout, seed))
    return SceneSpec(boxes=tuple(boxes), robot_start=layout.robot_start, cameras=cameras(layout))


@dataclass(frozen=True)
class Episode:
    """Which pool parts stand in the slots this time, where exactly, and what they truly are."""

    seed: int
    parts: tuple[str, ...]  # slot order
    poses: dict[str, tuple[tuple[float, float, float], float]]  # name -> ((x, y, z), yaw_deg)
    finishes: dict[str, Finish]
    bin_xy: tuple[tuple[str, float, float], ...] = ()  # where the bins truly are this episode (empty: where the layout says)

    def truth(self, name: str) -> Finish:
        return self.finishes[name]


def make_episode(layout: Layout, seed: int, scene_seed: int = 0, n_parts: int | None = None) -> Episode:
    """Deterministic for a seed: pick parts from the pool and jitter them into the slots."""
    rng = np.random.default_rng(seed)
    pool = part_pool(layout, scene_seed)
    n = len(layout.slot_xy) if n_parts is None else n_parts
    chosen = [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]
    poses = {}
    finishes = {}
    for slot, (name, finish) in enumerate(chosen):
        x, y, z = layout.slot_pos(slot)
        dx, dy = rng.uniform(-layout.jitter_xy_m, layout.jitter_xy_m, size=2)
        yaw = float(rng.uniform(-layout.jitter_yaw_deg, layout.jitter_yaw_deg))
        poses[name] = ((float(x + dx), float(y + dy), float(z)), yaw)
        finishes[name] = finish
    # The bins get their own random stream, so adding bin jitter left every seed's parts and poses as they were.
    bin_rng = np.random.default_rng([seed, 7919])
    bins = tuple(
        (name, float(cx + dx), float(cy + dy))
        for (name, cx, cy), (dx, dy) in zip(layout.bin_xy, bin_rng.uniform(-layout.bin_jitter_m, layout.bin_jitter_m, size=(len(layout.bin_xy), 2)), strict=True)
    )
    return Episode(seed, tuple(name for name, _ in chosen), poses, finishes, bins)


def episode_layout(layout: Layout, episode: Episode) -> Layout:
    """The layout as it truly is this episode: the bins where the episode put them. For ground truth, never for aiming."""
    return replace(layout, bin_xy=episode.bin_xy) if episode.bin_xy else layout


def place_episode(sim, layout: Layout, episode: Episode) -> None:
    """Put the episode's parts in their slots and every other pool part back in the park."""
    for i, name in enumerate(layout.part_names):
        if name in episode.poses:
            pos, yaw = episode.poses[name]
            sim.set_object_pose(name, pos, yaw)
        else:
            sim.set_object_pose(name, layout.park_pos(i), 0.0)
    true_layout = episode_layout(layout, episode)
    for name in layout.bin_names:
        for box in bin_boxes(true_layout, name):
            sim.set_object_pose(box.name, box.pos, 0.0)
