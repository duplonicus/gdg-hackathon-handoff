"""The soft-gel inspection cells as plain numbers (Nov 4; dup's own station from a capsule plant).

A **cell** is a work table with a tray of capsules on parchment, a person-tall rolling tray cart beside it, a small
leaker bin on the table and a big good bin on the floor. The job, in dup's words: one tray at a time, find the
leakers and pick them out, then dump the good capsules into the good bin. A **leaker** has a pool of its fill on the
parchment around it; a good capsule has none. The parchment is matte, so nothing else on it looks like a pool.

Two cells face each other across an aisle. The robot parks in front of one and drives to the other: "where am I
needed?" is its first decision. A finished cell shows an empty tray stack in its cart, a full good bin and a partly
full leaker bin.

Every cell is drawn in its own frame (the robot parked at the origin facing +x, the table ahead) and placed in the
world by ``Cell.to_world``. Sizes are a cafeteria tray (45 x 35 cm) and a cart about as tall as a person; capsules are
24 x 11 mm pills. Genesis carries a full tray at real time only when the capsules the robot moves are loose and the
rest are fixed (VALIDATION, capsule tray sizing), so ``loose`` names which ones move.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .sim.base import BoxSpec, CameraSpec

TABLE_COLOR = (0.50, 0.52, 0.55)  # brushed steel, drawn matte: no highlight from the tray lamps (dup, Nov 6: steel, no shine)
TRAY_COLOR = (0.62, 0.64, 0.66)  # stainless
PAPER_COLOR = (0.93, 0.89, 0.80)  # parchment
GEL_COLOR = (0.85, 0.62, 0.15)  # a fish-oil soft gel
POOL_COLOR = (0.80, 0.58, 0.12)  # its fill, spread thin on the paper
CART_COLOR = (0.45, 0.47, 0.50)
LEAKER_BIN_COLOR = (0.75, 0.20, 0.15)
GOOD_BIN_COLOR = (0.20, 0.45, 0.70)


BIN_SIDES = ("left", "right")  # the order of CellLayout.leaker_bins


@dataclass(frozen=True)
class CellLayout:
    """One cell in its own frame: metres, the robot parked at the origin facing +x.

    Why the table is open underneath (probe_body.py and probe_reach.py, Nov 4): the MH3's mobile base is 0.41 m tall and
    0.34 m ahead of its origin, its torso only 0.14 m. A table top on legs lets the base roll under it, as a person's
    legs go under a desk, until the torso is 2 cm from the edge.

    The station (Nov 5, dup: "we need to cut all of the driving"): from that one spot both hands together reach the
    whole tray, each its own half, turning the hand toward the middle where straight ahead falls short, and each
    reaches a leaker bin beside its half of the tray (probe_station.py: 86 of 87 sampled spots, every bin spot tried).
    The robot drives in once per shift and only its arms move until it backs out."""

    table_x: tuple[float, float] = (0.16, 0.76)  # the near edge 2 cm ahead of the torso with the base at the origin
    table_y: tuple[float, float] = (-0.90, 0.90)
    table_top: float = 0.82
    table_slab: float = 0.04
    table_leg: float = 0.05
    tray_size: tuple[float, float] = (0.35, 0.45)  # a cafeteria tray, x (deep) by y (wide)
    tray_center: tuple[float, float] = (0.355, 0.0)
    tray_rim: float = 0.015  # wall height above the tray floor
    tray_thick: float = 0.004
    paper_inset: float = 0.01  # parchment margin inside the tray walls
    capsule: tuple[float, float, float] = (0.024, 0.011, 0.011)  # length (x), width, height: a round pill
    # Capsule grid spacing, x (along each capsule) by y. Room for a pinch (Nov 5, grasp B): the index tip and thumb tip
    # close on a capsule's ends, so each end needs a fingertip's thickness (14 to 26 mm) clear; across, a fingertip
    # (11 to 17 mm wide) passes beside the next column. The Nov 4 tray was (0.030, 0.016): 260 capsules, too tight to pinch.
    pitch: tuple[float, float] = (0.065, 0.025)
    # Two half-trays, one per hand, with an empty aisle down the middle (Nov 5): palm down and straight ahead, neither
    # hand's pinch reaches within about 7 cm of the centre line on the near rows (probe_station.py --fresh, 49 of 64
    # spots of the full grid; output/cells/station_pinch.log). Every column kept is inside the measured reach.
    aisle: float = 0.15  # m of parchment left empty between the halves
    columns_per_side: int = 4
    # One per hand, beside its half of the tray (left, right)
    # (moved out to |y| 0.35 with the pinch: the wrist rides about 4 cm outboard of the capsule it picks)
    leaker_bins: tuple[tuple[float, float], tuple[float, float]] = ((0.30, 0.35), (0.30, -0.35))
    leaker_bin_inner: tuple[float, float] = (0.14, 0.14)
    leaker_bin_wall: float = 0.08
    # A dropped capsule falls about 10 cm and moves 14 mm a physics step: it went through a 1 cm floor into the table
    # (probe_pick_leaker.py, Nov 4). A 3 cm plate stops it.
    leaker_bin_floor: float = 0.03
    good_bin_center: tuple[float, float] = (0.45, -1.35)  # on the floor past the table's right end
    good_bin_inner: tuple[float, float] = (0.60, 0.50)  # wider than the tray, so a dump lands inside
    good_bin_wall: float = 0.55
    bin_thick: float = 0.01
    cart_center: tuple[float, float] = (0.45, 1.40)  # past the table's left end; its front (where trays slide) faces -x
    cart_size: tuple[float, float, float] = (0.42, 0.504, 1.75)  # depth (x), width (y): the rails hold a 45 cm tray by 1 cm a side
    cart_slots: int = 12
    cart_first_slot_z: float = 0.30

    def leaker_bin(self, side: str) -> tuple[float, float]:
        """The leaker bin the ``side`` hand ("left" or "right") drops into."""
        return self.leaker_bins[BIN_SIDES.index(side)]

    @property
    def paper_top(self) -> float:
        return self.table_top + self.tray_thick + 0.001

    @property
    def paper_half(self) -> tuple[float, float]:
        """Half the parchment's size: the area inside the tray walls where capsules lie."""
        return (self.tray_size[0] / 2.0 - self.paper_inset, self.tray_size[1] / 2.0 - self.paper_inset)

    def grid(self) -> list[tuple[float, float]]:
        """Every capsule spot on the parchment, row by row (cell frame)."""
        (sx, _), (px, py) = self.tray_size, self.pitch
        inner_x = sx - 2 * self.paper_inset - 0.01
        nx = int(inner_x // px)
        x0 = self.tray_center[0] - (nx - 1) * px / 2.0
        half = [self.aisle / 2.0 + py / 2.0 + k * py for k in range(self.columns_per_side)]
        ys = sorted([self.tray_center[1] - d for d in half] + [self.tray_center[1] + d for d in half])
        return [(x0 + i * px, y) for y in ys for i in range(nx)]

    def cart_slot_z(self, slot: int) -> float:
        top = self.cart_size[2] - 0.10
        step = (top - self.cart_first_slot_z) / max(1, self.cart_slots - 1)
        return self.cart_first_slot_z + slot * step


# Where the left hand works, relative to the base (probe_reach.py, wrist 0.88 m, Nov 4): its wrist reaches the whole
# side band at 0.25 to 0.40 m ahead, and the hand's working point is about 0.12 m ahead of the wrist (fake.HOLD_OFFSET).
HAND_AHEAD = (0.37, 0.52)  # m ahead of the base origin where the left hand covers its whole side band
HAND_SIDE = 0.25  # m to the left of the base origin: the middle of that band (wrist 0.10 to 0.50, hand 4.5 cm inboard)
BASE_HALF = (0.34, 0.29)  # the mobile base's half length and half width (probe_body.py)


def park_for(layout: CellLayout, x: float, y: float) -> tuple[float, float]:
    """Where the base stands (cell frame, facing +x) for the left hand to work the point (x, y) on the table.

    The Nov 4 one-hand layout, kept for its probes (probe_drive_cells.py, probe_pick_leaker.py); the task parks once
    at the station since Nov 5 (capsule_skills.park_station).

    As close as the torso allows (base origin no further than the cell origin), backed off when the point is nearer
    than the hand reaches, and sideways so the point sits in the middle of the hand's band. Raises if the legs are in
    the way: the base fits between them only so far."""
    bx = min(0.0, x - sum(HAND_AHEAD) / 2.0)
    by = y - HAND_SIDE
    leg_room = layout.table_y[1] - layout.table_leg - BASE_HALF[1] - 0.02
    if abs(by) > leg_room:
        raise ValueError(f"({x:.3f}, {y:.3f}) needs the base at y {by:.3f}, past the table legs (+-{leg_room:.3f})")
    if not HAND_AHEAD[0] <= x - bx <= HAND_AHEAD[1]:
        raise ValueError(f"({x:.3f}, {y:.3f}) is {x - bx:.3f} m ahead of the closest base spot, outside {HAND_AHEAD}")
    return (float(bx), float(by))


@dataclass(frozen=True)
class Cell:
    """A cell placed in the world: its name, where its park pose is, and which way it faces."""

    name: str
    origin: tuple[float, float] = (0.0, 0.0)
    yaw_deg: float = 0.0
    layout: CellLayout = field(default_factory=CellLayout)

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        return (float(self.origin[0] + c * x - s * y), float(self.origin[1] + s * x + c * y))

    def place(self, box: BoxSpec) -> BoxSpec:
        """A box drawn in the cell frame, moved into the world and named for this cell."""
        x, y = self.to_world(box.pos[0], box.pos[1])
        return replace(box, name=f"{self.name}_{box.name}", pos=(x, y, box.pos[2]), yaw_deg=box.yaw_deg + self.yaw_deg)

    @property
    def park(self) -> tuple[float, float, float]:
        """Where the robot's base stands to work this cell: x, y, yaw in degrees."""
        return (*self.origin, self.yaw_deg)


def _open_box(name: str, center: tuple[float, float], floor_z: float, inner: tuple[float, float], wall: float, thick: float, color, floor: float | None = None) -> list[BoxSpec]:
    """A floor plate (``floor`` thick, default the wall thickness) and four walls around an inner footprint, all fixed."""
    (cx, cy), (ix, iy), t = center, inner, thick
    f = t if floor is None else floor
    ox, oy = ix + 2 * t, iy + 2 * t
    zf, zw = floor_z + f / 2.0, floor_z + f + wall / 2.0
    return [
        BoxSpec(f"{name}_floor", (ox, oy, f), (cx, cy, zf), color=color, fixed=True),
        BoxSpec(f"{name}_wall_xm", (t, oy, wall), (cx - (ix + t) / 2.0, cy, zw), color=color, fixed=True),
        BoxSpec(f"{name}_wall_xp", (t, oy, wall), (cx + (ix + t) / 2.0, cy, zw), color=color, fixed=True),
        BoxSpec(f"{name}_wall_ym", (ix, t, wall), (cx, cy - (iy + t) / 2.0, zw), color=color, fixed=True),
        BoxSpec(f"{name}_wall_yp", (ix, t, wall), (cx, cy + (iy + t) / 2.0, zw), color=color, fixed=True),
    ]


def cell_furniture(layout: CellLayout) -> list[BoxSpec]:
    """Table, tray, parchment, both bins and the cart, in the cell frame. Nothing here moves."""
    lay = layout
    (x0, x1), (y0, y1) = lay.table_x, lay.table_y
    slab, leg = lay.table_slab, lay.table_leg
    boxes = [BoxSpec("table", (x1 - x0, y1 - y0, slab), ((x0 + x1) / 2.0, (y0 + y1) / 2.0, lay.table_top - slab / 2.0), color=TABLE_COLOR, fixed=True, look="matte")]
    for sx, lx in (("n", x0 + leg / 2.0), ("f", x1 - leg / 2.0)):
        for sy, ly in (("r", y0 + leg / 2.0), ("l", y1 - leg / 2.0)):
            boxes.append(BoxSpec(f"table_leg_{sx}{sy}", (leg, leg, lay.table_top - slab), (lx, ly, (lay.table_top - slab) / 2.0), color=TABLE_COLOR, fixed=True, look="matte"))
    tray_inner = (lay.tray_size[0] - 2 * lay.tray_thick, lay.tray_size[1] - 2 * lay.tray_thick)
    boxes += _open_box("tray", lay.tray_center, lay.table_top, tray_inner, lay.tray_rim, lay.tray_thick, TRAY_COLOR)
    paper = (lay.tray_size[0] - 2 * lay.paper_inset, lay.tray_size[1] - 2 * lay.paper_inset, 0.001)
    boxes.append(BoxSpec("parchment", paper, (*lay.tray_center, lay.paper_top - 0.0005), color=PAPER_COLOR, fixed=True, look="matte"))
    for side in BIN_SIDES:
        boxes += _open_box(f"leaker_bin_{side}", lay.leaker_bin(side), lay.table_top, lay.leaker_bin_inner, lay.leaker_bin_wall, lay.bin_thick, LEAKER_BIN_COLOR, floor=lay.leaker_bin_floor)
    boxes += _open_box("good_bin", lay.good_bin_center, 0.0, lay.good_bin_inner, lay.good_bin_wall, lay.bin_thick, GOOD_BIN_COLOR)
    boxes += cart_boxes(lay)
    return boxes


def cart_boxes(layout: CellLayout) -> list[BoxSpec]:
    """The rolling tray cart: four corner posts, a top, a base, and a pair of rails per tray slot. Open at the front (-x)."""
    (cx, cy), (dx, dy, h) = layout.cart_center, layout.cart_size
    post, rail = 0.025, 0.012
    boxes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            tall = h - 0.10 - 0.02  # from the base's top to the top plate's underside
            boxes.append(BoxSpec(f"cart_post_{'f' if sx < 0 else 'b'}{'r' if sy < 0 else 'l'}", (post, post, tall), (cx + sx * (dx - post) / 2.0, cy + sy * (dy - post) / 2.0, 0.10 + tall / 2.0), color=CART_COLOR, fixed=True))
    boxes.append(BoxSpec("cart_base", (dx, dy, 0.03), (cx, cy, 0.085), color=CART_COLOR, fixed=True))
    boxes.append(BoxSpec("cart_top", (dx, dy, 0.02), (cx, cy, h - 0.01), color=CART_COLOR, fixed=True))
    inner_y = dy - 2 * post
    for slot in range(layout.cart_slots):
        z = layout.cart_slot_z(slot) - rail / 2.0
        for side, sy in (("r", -1), ("l", 1)):
            boxes.append(BoxSpec(f"cart_rail_{slot:02d}{side}", (dx - 2 * post, rail, rail), (cx, cy + sy * (inner_y - rail) / 2.0, z), color=CART_COLOR, fixed=True))
    return boxes


def capsule_boxes(layout: CellLayout, leakers: frozenset[int], loose: frozenset[int]) -> list[BoxSpec]:
    """Every capsule on the tray (cell frame) and a pool under each leaker. Only ``loose`` ones are free bodies."""
    lay = layout
    length, width, height = lay.capsule
    boxes = []
    for i, (x, y) in enumerate(lay.grid()):
        boxes.append(BoxSpec(f"capsule_{i:03d}", lay.capsule, (x, y, lay.paper_top + height / 2.0), color=GEL_COLOR, rho=1100.0, shape="capsule", look="glossy", fixed=i not in loose))
        if i in leakers:
            boxes.append(BoxSpec(f"pool_{i:03d}", (0.030, 0.030, 0.0006), (x, y, lay.paper_top + 0.0003), color=POOL_COLOR, fixed=True, shape="disc", look="glossy", collision=False))
    return boxes


def cell_camera(cell: Cell) -> CameraSpec:
    """A fixed camera over the aisle that sees the whole cell: cart, table, tray and both bins."""
    lay = cell.layout
    eye = cell.to_world(-0.55, 0.10)
    look = cell.to_world(0.60, 0.15)
    return CameraSpec(f"{cell.name}_overview", pos=(*eye, 2.10), lookat=(*look, 0.60), fov=70, res=(640, 400))


def tray_camera(cell: Cell) -> CameraSpec:
    """The inspection camera over the tray, mounted above its far side looking back and down: the robot works from
    the near side, and a camera on that side had the robot's raised forearm across half the tray (Nov 4)."""
    lay = cell.layout
    cx, cy = lay.tray_center
    # 1920 x 1200: at 960 x 600 a capsule was 23 pixels long and a block's picture 45 x 95 (Nov 4 smoke run)
    return CameraSpec(f"{cell.name}_tray", pos=(*cell.to_world(cx + 0.18, cy), lay.table_top + 0.55), lookat=(*cell.to_world(cx, cy), lay.table_top), fov=45, res=(1920, 1200))


def bin_camera(cell: Cell, side: str) -> CameraSpec:
    """A camera over one leaker bin, for "did the capsule land in it": high and nearly straight down, so the bin's
    floor is in view past its walls (a lower, slanted one saw mostly wall and shadow, Nov 4 smoke run)."""
    lay = cell.layout
    bx, by = lay.leaker_bin(side)
    return CameraSpec(f"{cell.name}_leaker_bin_{side}", pos=(*cell.to_world(bx + 0.06, by), lay.table_top + 0.60), lookat=(*cell.to_world(bx, by), lay.table_top), fov=30, res=(480, 300))


def overlapping(boxes: list[BoxSpec], tol: float = 1e-4) -> list[tuple[str, str, float]]:
    """Every pair of solid boxes whose volumes overlap by more than ``tol`` metres on every axis.

    Each box's footprint is its yaw-rotated rectangle's bounding box, which over-reports for boxes turned off 0/90/180
    degrees (a false alarm, never a miss). Touching faces (resting on, leaning against) do not count. Drawn-only boxes
    (``collision=False``: pools) are decals and skipped."""
    solid = [b for b in boxes if b.collision]
    lo, hi = [], []
    for b in solid:
        c, s = abs(np.cos(np.radians(b.yaw_deg))), abs(np.sin(np.radians(b.yaw_deg)))
        half = np.array([(c * b.size[0] + s * b.size[1]) / 2.0, (s * b.size[0] + c * b.size[1]) / 2.0, b.size[2] / 2.0])
        lo.append(np.array(b.pos) - half)
        hi.append(np.array(b.pos) + half)
    found = []
    for i in range(len(solid)):
        for j in range(i + 1, len(solid)):
            depth = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])
            if (depth > tol).all():
                found.append((solid[i].name, solid[j].name, float(depth.min())))
    return found


FLOOR_TILE = 0.5  # m: the tile pitch of the capsule cells' floor
FLOOR_SIZE = 24.0  # m square, round the two cells; past it the plane still collides but is not drawn


def floor_texture(tiles: int, px: int = 32, seed: int = 0) -> np.ndarray:
    """White-grey floor tiles with darker grout, each tile a shade apart (dup, Nov 6: a whitish grey tile floor in place
    of the kit's blue checker)."""
    rng = np.random.default_rng(seed)
    shade = 0.84 + rng.uniform(-0.015, 0.015, size=(tiles, tiles))
    img = np.repeat(np.repeat(shade, px, axis=0), px, axis=1)
    grout = np.zeros((px, px), dtype=bool)
    grout[:1, :] = grout[:, :1] = True
    img[np.tile(grout, (tiles, tiles))] = 0.66
    return (np.repeat(img[:, :, None], 3, axis=2) * 255).astype(np.uint8)


GROUND_SIZE = 120.0  # m square: the plain ground round the tiles, over the park 40 m aside and the offstage spot
GROUND_COLOR = (0.82, 0.82, 0.82)  # the tiles' shade with their grout, seen from afar


def floor_box(center: tuple[float, float]) -> BoxSpec:
    """The drawn floor: a thin slab whose top is the kit's plane, drawn only (the plane does the physics)."""
    tiles = int(FLOOR_SIZE / FLOOR_TILE)
    return BoxSpec("floor_tiles", (FLOOR_SIZE, FLOOR_SIZE, 0.01), (*center, -0.005), fixed=True, collision=False, texture=floor_texture(tiles))


def ground_box() -> BoxSpec:
    """Plain ground 2 cm under the tiles and far past them: with the kit's plane no longer drawn, it hides what waits
    under the floor in the park 40 m aside (the reason for PARK_Z), seen from a viewer turned that way."""
    return BoxSpec("floor_ground", (GROUND_SIZE, GROUND_SIZE, 0.01), (0.0, 0.0, -0.025), color=GROUND_COLOR, fixed=True, collision=False, look="matte")


def pill_texture(rows: int, cols: int, px: int = 8, fill: float = 1.0, seed: int = 0) -> np.ndarray:
    """A top-down picture of capsules on parchment, ``fill`` of the spots taken: the tops of full cart trays and bin heaps."""
    rng = np.random.default_rng(seed)
    h, w = rows * px, cols * px * 2
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = np.array(PAPER_COLOR) * 255
    gel = (np.array(GEL_COLOR) * 255).astype(np.uint8)
    yy, xx = np.mgrid[0:px, 0:2 * px]
    pill = ((xx - px + 0.5) / (px - 1.0)) ** 8 + ((yy - px / 2 + 0.5) / (px / 2 - 1.0)) ** 2 <= 1.0
    for r in range(rows):
        for c in range(cols):
            if rng.random() < fill:
                img[r * px:(r + 1) * px, c * 2 * px:(c + 1) * 2 * px][pill] = gel
    return img


TRAY_IN_CART = (0.35, 0.45, 0.012)  # a tray turned to slide in: 35 cm deep (x), 45 cm wide (y)


def cart_trays(layout: CellLayout, full: int) -> list[BoxSpec]:
    """A tray on every cart slot, the bottom ``full`` of them still holding capsules (a picture of them on top)."""
    (cx, cy) = layout.cart_center
    boxes = []
    for slot in range(layout.cart_slots):
        z = layout.cart_slot_z(slot) + TRAY_IN_CART[2] / 2.0
        if slot < full:
            boxes.append(BoxSpec(f"cart_tray_{slot:02d}", TRAY_IN_CART, (cx, cy, z), fixed=True, texture=pill_texture(22, 11, seed=slot)))
        else:
            boxes.append(BoxSpec(f"cart_tray_{slot:02d}", TRAY_IN_CART, (cx, cy, z), color=PAPER_COLOR, fixed=True, look="matte"))
    return boxes


def bin_fill(name: str, center: tuple[float, float], floor_top: float, inner: tuple[float, float], level: float, seed: int = 0) -> BoxSpec:
    """The heap in a bin drawn as a block level with its top: ``level`` metres deep (0: an empty bin, no block)."""
    depth = max(level, 0.002)
    fill = 1.0 if level > 0.0 else 0.0
    return BoxSpec(f"{name}_heap", (inner[0] - 0.002, inner[1] - 0.002, depth), (*center, floor_top + depth / 2.0), fixed=True, texture=pill_texture(30, 20, fill=fill, seed=seed))


# ----- the two-cell scene and its episodes -----
#
# Genesis builds a scene once, so everything any episode needs exists from the start and an episode only moves things:
# - one grid of FIXED capsules, standing in whichever cell has work this episode (parked under the floor otherwise);
# - a few LOOSE capsules for the leakers, which the robot picks: each takes the place of the fixed capsule at its spot;
# - a pool disc per loose capsule, drawn only;
# - per cell, both looks of its cart and bins (with work waiting, and finished), the unused look parked out of sight.

MAX_LEAKERS = 5
MAX_SPARES = 5  # loose good capsules, swapped in for a fixed one the robot decides to pick (a false alarm)
# Pool diameters, from a fresh leak that barely shows past the capsule's 11 mm width to a spread one. A disc's size is
# fixed when the scene is built, so every leaker slot gets one disc of each size and an episode shows one of them.
POOL_SIZES = (0.016, 0.022, 0.030)
PARK_Z = -2.0  # under the floor: out of every camera and every contact
# ... and 40 m to the side, out of the viewer too: the floor is one plane and the viewer saw parked props through it
# from below (dup, Nov 6: "why are there objects under the floor?"). Genesis cannot add or remove bodies after build,
# so everything a shift might need is built once and whatever is not in use waits here.
PARK_DX = 40.0
OFFSTAGE = (40.0, 40.0)  # where the loose leakers and spares wait, resting on the floor far from everything


def parked_pos(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Where a parked thing waits: the same layout shifted 40 m aside and 2 m under the floor."""
    return (x + PARK_DX, y, PARK_Z + z)
LAMP_HEIGHT = 1.2  # m above the table top
LAMP_INTENSITY = 3.0
AMBIENT = 0.3


@dataclass(frozen=True)
class TwoCells:
    """Two cells facing each other across an aisle; the robot spawns in the aisle facing cell a.

    The kit spawns the robot at x -1.2 (config/handoff.yaml), so the aisle is centred there."""

    layout: CellLayout = field(default_factory=CellLayout)
    a_origin: tuple[float, float] = (-0.80, 0.0)
    b_origin: tuple[float, float] = (-1.60, 0.0)

    @property
    def cells(self) -> tuple[Cell, Cell]:
        return (Cell("a", self.a_origin, 0.0, self.layout), Cell("b", self.b_origin, 180.0, self.layout))

    def cell(self, name: str) -> Cell:
        return {c.name: c for c in self.cells}[name]

    @property
    def grid_names(self) -> tuple[str, ...]:
        return tuple(f"capsule_{i:03d}" for i in range(len(self.layout.grid())))

    @property
    def leaker_names(self) -> tuple[str, ...]:
        return tuple(f"leaker_{k}" for k in range(MAX_LEAKERS))

    @property
    def spare_names(self) -> tuple[str, ...]:
        return tuple(f"spare_{k}" for k in range(MAX_SPARES))

    @staticmethod
    def pool_name(k: int, size: int) -> str:
        return f"pool_{k}_{size}"

    @property
    def pool_names(self) -> tuple[str, ...]:
        """Each leaker slot's biggest pool: what a probe that places one pool per leaker uses."""
        return tuple(self.pool_name(k, len(POOL_SIZES) - 1) for k in range(MAX_LEAKERS))

    @property
    def all_pool_names(self) -> tuple[str, ...]:
        return tuple(self.pool_name(k, size) for k in range(MAX_LEAKERS) for size in range(len(POOL_SIZES)))


FULL_LOOK = {"cart_full": 11, "good_level": 0.04, "leaker_level": 0.0}  # work waiting: one tray out on the table
DONE_LOOK = {"cart_full": 0, "good_level": 0.42, "leaker_level": 0.012}  # finished: empty trays, good bin full


def look_boxes(cell: Cell, look: str) -> list[BoxSpec]:
    """One look of a cell's cart trays and bin heaps, in the world, named ``<cell>_<look>_...``."""
    lay = cell.layout
    numbers = FULL_LOOK if look == "work" else DONE_LOOK
    boxes = [replace(b, name=f"{look}_{b.name}") for b in cart_trays(lay, numbers["cart_full"])]
    if numbers["good_level"] > 0:
        boxes.append(bin_fill(f"{look}_good_bin", lay.good_bin_center, lay.bin_thick, lay.good_bin_inner, numbers["good_level"], seed=1))
    if numbers["leaker_level"] > 0:
        for k, side in enumerate(BIN_SIDES):
            boxes.append(bin_fill(f"{look}_leaker_bin_{side}", lay.leaker_bin(side), lay.table_top + lay.leaker_bin_floor, lay.leaker_bin_inner, numbers["leaker_level"], seed=2 + k))
    return [cell.place(b) for b in boxes]


def parked(box: BoxSpec) -> BoxSpec:
    """The same box moved straight down under the floor, where no camera or contact sees it."""
    return replace(box, pos=parked_pos(*box.pos))


def visible(boxes) -> list[BoxSpec]:
    """The boxes above the floor: what a camera can see and a robot can touch."""
    return [b for b in boxes if b.pos[2] + b.size[2] / 2.0 > 0.0]


def make_two_cell_scene(scene: TwoCells | None = None):
    """Everything built once, in the poses of episode zero's layout: cell a has work, cell b is finished."""
    from .sim.base import SceneSpec

    scene = scene or TwoCells()
    a, b = scene.cells
    boxes: list[BoxSpec] = [floor_box(((a.origin[0] + b.origin[0]) / 2.0, (a.origin[1] + b.origin[1]) / 2.0)), ground_box()]
    for cell, shown in ((a, "work"), (b, "done")):
        boxes += [cell.place(x) for x in cell_furniture(cell.layout)]
        for look in ("work", "done"):
            boxes += [x if look == shown else parked(x) for x in look_boxes(cell, look)]
    lay = scene.layout
    for i, (x, y) in enumerate(lay.grid()):
        wx, wy = a.to_world(x, y)
        boxes.append(BoxSpec(scene.grid_names[i], lay.capsule, (wx, wy, lay.paper_top + lay.capsule[2] / 2.0), yaw_deg=a.yaw_deg, color=GEL_COLOR, rho=1100.0, shape="capsule", look="glossy", fixed=True))
    for k in range(MAX_LEAKERS):
        boxes.append(BoxSpec(scene.leaker_names[k], lay.capsule, (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1], lay.capsule[2] / 2.0), color=GEL_COLOR, rho=1100.0, friction=1.5, shape="capsule", look="glossy"))
        for size, diameter in enumerate(POOL_SIZES):
            boxes.append(BoxSpec(scene.pool_name(k, size), (diameter, diameter, 0.0006), (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1] + 0.2 + 0.1 * size, PARK_Z), color=POOL_COLOR, fixed=True, shape="disc", look="glossy", collision=False))
    for k in range(MAX_SPARES):
        boxes.append(BoxSpec(scene.spare_names[k], lay.capsule, (OFFSTAGE[0] + 0.1 * k, OFFSTAGE[1] - 0.4, lay.capsule[2] / 2.0), color=GEL_COLOR, rho=1100.0, friction=1.5, shape="capsule", look="glossy"))
    cams = tuple(cam for cell in (a, b) for cam in (cell_camera(cell), tray_camera(cell), *(bin_camera(cell, side) for side in BIN_SIDES)))
    # Lamps over each tray, as an inspection station has (Nov 4: with only the kit's sun, the robot's shadow lay across
    # half of cell b's tray). Two lamps a tray, one over each half, and no sun (dup, Nov 6): the kit's low directional sun threw the robot's arm
    # across the tray in diagonal shade bands, and d3 called 62% of shaded clean capsules leakers against 15% lit ones
    # (probe_tray_light.py, VALIDATION "What casts the shade"). An inspection station is lit from above, not by a sun.
    cx, cy = lay.tray_center
    lamps = tuple((*cell.to_world(cx, cy + dy), lay.table_top + LAMP_HEIGHT, LAMP_INTENSITY / 2.0) for cell in (a, b) for dy in (-0.25, 0.25))
    return SceneSpec(boxes=tuple(boxes), robot_start=(-1.2, 0.0, 0.0), cameras=cams, lamps=lamps, ambient=AMBIENT, sun=0.0, kit_floor=False)
