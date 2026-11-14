"""Finding capsules in a synthetic point cloud (Nov 4): height above the paper, never colour, picks them out."""

from __future__ import annotations

import numpy as np

from handoff.capsule_perception import crop_numbered, find_capsules

PAPER = 0.825


def cloud(capsules, pools=(), size=(120, 200), mm_per_px=2.0, origin=(0.30, -0.20)):
    """A top-down camera: pixel (r, c) sees world (origin x + r * mm, origin y + c * mm). Paper everywhere; each
    capsule a raised rectangle (length along x, width along y); each pool a 0.6 mm film disc."""
    rows, cols = np.mgrid[0:size[0], 0:size[1]]
    x = origin[0] + rows * mm_per_px / 1000.0
    y = origin[1] + cols * mm_per_px / 1000.0
    z = np.full(x.shape, PAPER)
    for px, py, r in pools:
        z = np.where(np.hypot(x - px, y - py) <= r, PAPER + 0.0006, z)
    for cx, cy in capsules:
        z = np.where((np.abs(x - cx) <= 0.012) & (np.abs(y - cy) <= 0.0055), PAPER + 0.011, z)
    return np.stack([x, y, z], axis=-1)


TRAY = ((0.42, 0.0), (0.12, 0.20))


def test_finds_every_capsule_where_it_is_and_in_tray_order():
    spots = [(0.40, -0.10), (0.40, 0.05), (0.43, -0.10), (0.43, 0.05)]
    found = find_capsules(cloud(spots), PAPER, *TRAY)
    assert len(found) == 4
    got = [(round(c.xyz[0], 3), round(c.xyz[1], 3)) for c in found]
    for (gx, gy), (sx, sy) in zip(got, sorted(spots), strict=True):
        assert abs(gx - sx) <= 0.002 and abs(gy - sy) <= 0.002


def test_a_pool_is_not_a_capsule_and_does_not_merge_neighbours():
    """A leaker's pool spreads under its neighbours; height still separates three capsules from the film."""
    spots = [(0.42, -0.016), (0.42, 0.0), (0.42, 0.016)]  # 5 mm gaps, like the tray grid
    found = find_capsules(cloud(spots, pools=[(0.42, 0.0, 0.015)]), PAPER, *TRAY)
    assert len(found) == 3


def test_ignores_anything_off_the_tray():
    found = find_capsules(cloud([(0.42, 0.0), (0.42, 0.18)]), PAPER, (0.42, 0.0), (0.12, 0.10))  # second one is past the tray's edge
    assert len(found) == 1


def test_a_turned_tray_still_bounds_the_search():
    # the tray turned 90 degrees: its long side now runs along world x
    found = find_capsules(cloud([(0.42, 0.0), (0.50, 0.0)]), PAPER, (0.42, 0.0), (0.20, 0.03), tray_yaw_deg=90.0)
    assert len(found) == 1  # (0.50, 0) is 8 cm along world x: inside a 3 cm half width once turned


def test_crop_holds_every_capsule_and_its_numbers():
    spots = [(0.40, -0.10), (0.40, 0.05)]
    xyz = cloud(spots)
    rgb = np.zeros((*xyz.shape[:2], 3), dtype=np.uint8)
    found = find_capsules(xyz, PAPER, *TRAY)
    crop = crop_numbered(rgb, found, pad=5)
    top = min(c.box[0] for c in found) - 5
    left = min(c.box[1] for c in found) - 5
    assert crop.shape[0] == max(c.box[2] for c in found) + 5 - max(top, 0)
    assert crop.shape[1] == max(c.box[3] for c in found) + 5 - max(left, 0)
    assert (crop == 255).any()  # the number labels' white boxes


def test_the_tray_walls_outside_the_search_are_not_a_capsule():
    """A tray rim 14 mm above the paper, just outside the parchment: searching the parchment's half size leaves it out."""
    xyz = cloud([(0.42, 0.0)])
    rim = (np.abs(xyz[..., 1] - 0.15) <= 0.004)  # a wall along x at y 0.15
    xyz[..., 2] = np.where(rim, PAPER + 0.014, xyz[..., 2])
    assert len(find_capsules(xyz, PAPER, (0.42, 0.0), (0.12, 0.20))) == 2  # searching up to the wall's outside: rim counted
    assert len(find_capsules(xyz, PAPER, (0.42, 0.0), (0.12, 0.14))) == 1  # the parchment only


def test_a_column_on_a_rounding_edge_stays_one_column():
    """Capsules of one column whose x straddles a 1 cm step (15 mm, as the real grid's columns do) stay together."""
    from handoff.capsule_perception import SeenCapsule, in_rows

    caps = [SeenCapsule((0.0149 if k % 2 else 0.0151, 0.016 * k, 0.83), (0, 0, 1, 1), 40) for k in range(6)]
    caps += [SeenCapsule((0.045, 0.016 * k, 0.83), (0, 0, 1, 1), 40) for k in range(6)]
    ordered = in_rows(caps, (0.0, 0.0), 1.0, 0.0)
    assert [round(c.xyz[0], 3) for c in ordered[:6]] == [0.015] * 6  # the first column whole, before the second
    assert [c.xyz[1] for c in ordered[:6]] == sorted(c.xyz[1] for c in ordered[:6])
