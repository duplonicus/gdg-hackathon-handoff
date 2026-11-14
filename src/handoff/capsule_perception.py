"""Finding the capsules on a tray from the inspection camera's colour image and point cloud (Nov 4).

A capsule stands 11 mm proud of the parchment; a pool of fill is a film under a millimetre thick. So every pixel the
camera sees more than 3 mm above the paper, inside the tray, is capsule, and each connected patch of such pixels is
one capsule. This is plain geometry, the kind of step a cell's vision system does without any model: it says where
the capsules are, never which ones leak. That judgement is the model's (or a person's), from the picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABOVE_PAPER = 0.003  # m: higher than any pool, lower than any capsule's top
MIN_PIXELS = 30  # a patch smaller than this is a sliver of a capsule mostly hidden, or noise


@dataclass(frozen=True)
class SeenCapsule:
    """One capsule as the camera found it: where it is in the world, and the pixels it covers."""

    xyz: tuple[float, float, float]  # the mean of its points: its top surface's middle, near enough
    box: tuple[int, int, int, int]  # pixel rows and columns it spans: (top, left, bottom, right), bottom/right exclusive
    pixels: int


def find_capsules(xyz: np.ndarray, paper_top: float, tray_xy: tuple[float, float], tray_half: tuple[float, float], tray_yaw_deg: float = 0.0) -> list[SeenCapsule]:
    """Every capsule the camera sees on the tray, ordered along the tray (its x, then its y).

    :param xyz: (H, W, 3) world position of what each pixel sees.
    :param paper_top: the parchment's top, m.
    :param tray_xy, tray_half, tray_yaw_deg: the tray's middle, heading, and the half size of the parchment inside its
        walls (the walls stand higher than the paper too, and the first Genesis check counted them as one more capsule).
    """
    from scipy import ndimage

    c, s = np.cos(np.radians(tray_yaw_deg)), np.sin(np.radians(tray_yaw_deg))
    dx, dy = xyz[..., 0] - tray_xy[0], xyz[..., 1] - tray_xy[1]
    u, v = c * dx + s * dy, -s * dx + c * dy  # in the tray's own frame
    on_tray = (np.abs(u) <= tray_half[0]) & (np.abs(v) <= tray_half[1])
    height = xyz[..., 2] - paper_top
    mask = on_tray & (height > ABOVE_PAPER) & (height < 0.03)
    labels, count = ndimage.label(mask)
    found = []
    for index, rows_cols in enumerate(ndimage.find_objects(labels), start=1):
        if rows_cols is None:
            continue
        mine = labels[rows_cols] == index
        n = int(mine.sum())
        if n < MIN_PIXELS:
            continue
        points = xyz[rows_cols][mine]
        rows, cols = rows_cols
        found.append(SeenCapsule(tuple(float(v) for v in points.mean(axis=0)), (rows.start, cols.start, rows.stop, cols.stop), n))
    return in_rows(found, tray_xy, c, s)


def in_rows(found: list[SeenCapsule], tray_xy, c: float, s: float, gap: float = 0.01) -> list[SeenCapsule]:
    """Order capsules along the tray's x, then its y: a new column starts wherever the next capsule along x is more than
    ``gap`` further on. Rounding x to a fixed step split columns that sat on a step's edge (commit review, Nov 4)."""
    if not found:
        return []
    u = lambda cap: c * (cap.xyz[0] - tray_xy[0]) + s * (cap.xyz[1] - tray_xy[1])
    v = lambda cap: -s * (cap.xyz[0] - tray_xy[0]) + c * (cap.xyz[1] - tray_xy[1])
    along = sorted(found, key=u)
    columns, column = [], [along[0]]
    for prev, cap in zip(along, along[1:], strict=False):
        if u(cap) - u(prev) > gap:
            columns.append(column)
            column = []
        column.append(cap)
    columns.append(column)
    return [cap for col in columns for cap in sorted(col, key=v)]


def crop_numbered(rgb: np.ndarray, capsules: list[SeenCapsule], pad: int = 12) -> np.ndarray:
    """The picture around these capsules, each with its number (1, 2, ...) written beside it: what a model is shown
    when asked which of them leaks. The crop spans every capsule's pixels plus ``pad`` on each side, so a pool around
    an edge capsule is in view."""
    from PIL import Image, ImageDraw

    top = max(0, min(cap.box[0] for cap in capsules) - pad)
    left = max(0, min(cap.box[1] for cap in capsules) - pad)
    bottom = min(rgb.shape[0], max(cap.box[2] for cap in capsules) + pad)
    right = min(rgb.shape[1], max(cap.box[3] for cap in capsules) + pad)
    image = Image.fromarray(np.ascontiguousarray(rgb[top:bottom, left:right]))
    draw = ImageDraw.Draw(image)
    for number, cap in enumerate(capsules, start=1):
        y, x = cap.box[0] - top, cap.box[1] - left
        draw.rectangle((x - 1, y - 1, x + 13, y + 11), fill=(255, 255, 255))
        draw.text((x + 1, y), str(number), fill=(0, 0, 0))
    return np.asarray(image)
