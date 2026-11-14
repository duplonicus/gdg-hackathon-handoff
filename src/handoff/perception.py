"""What the robot works out from camera data instead of being told.

locate_bins finds each bin tray in a colour image with a point per pixel (the fixed camera over the bins): the
pixels of the tray's colour at the height of its wall tops form a square ring, and the ring's middle is the bin.
Parts inside a bin are another colour, and the table, though close to orange, is 9 cm lower.
"""

from __future__ import annotations

import numpy as np

from .scene import Layout


def _is_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (g > 1.25 * r) & (g > 1.25 * b) & (g > 40)


def _is_orange(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (r > 1.2 * g) & (g > 1.5 * b) & (r > 60)


BIN_COLOR_TESTS = {"good": _is_green, "rework": _is_orange}


def locate_bins(rgb: np.ndarray, xyz: np.ndarray, layout: Layout, min_points: int = 40) -> dict[str, tuple[float, float]]:
    """Bin name -> (x, y) of its centre in the world, for every bin found. A bin that is not found is left out.

    :param rgb: (H, W, 3) uint8 colour image.
    :param xyz: (H, W, 3) world position of what each pixel sees.
    """
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    wall_top = layout.table_top + layout.bin_thick + layout.bin_wall_h
    at_rim = np.abs(xyz[..., 2] - wall_top) < 0.012
    outer = layout.bin_inner + 2 * layout.bin_thick
    found: dict[str, tuple[float, float]] = {}
    for name in layout.bin_names:
        points = xyz[BIN_COLOR_TESTS[name](r, g, b) & at_rim]
        if len(points) < min_points:
            continue
        x0, x1 = np.percentile(points[:, 0], [1, 99])
        y0, y1 = np.percentile(points[:, 1], [1, 99])
        # A whole rim spans the tray's outer size both ways; anything else means part of it was hidden.
        if abs((x1 - x0) - outer) > 0.03 or abs((y1 - y0) - outer) > 0.03:
            continue
        found[name] = (float((x0 + x1) / 2.0), float((y0 + y1) / 2.0))
    return found
