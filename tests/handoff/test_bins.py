"""The robot finds the bins from the camera; the scene file only says roughly where they are."""

from pathlib import Path

import numpy as np

from handoff.decider import Ladder, Step
from handoff.logbook import Logbook
from handoff.perception import locate_bins
from handoff.rungs import OracleRung
from handoff.scene import Layout, episode_layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import TRAVEL_POSE
from handoff.tasks.sort import SortContext, run_sort, truth
import handoff.tasks.sort as sort_module


def rim_cloud(layout, centres, table=True):
    """A synthetic view: each tray's rim in its colour, plus a brown table 9 cm below that must not be mistaken for it."""
    colors, points = [], []
    half = layout.bin_inner / 2 + layout.bin_thick
    top = layout.table_top + layout.bin_thick + layout.bin_wall_h
    for name, (cx, cy) in centres.items():
        rgb = (51, 153, 51) if name == "good" else (204, 128, 26)
        for t in np.linspace(-half, half, 60):
            for x, y in ((cx + t, cy - half), (cx + t, cy + half), (cx - half, cy + t), (cx + half, cy + t)):
                points.append((x, y, top))
                colors.append(rgb)
    if table:
        for x in np.linspace(0.3, 0.8, 40):
            for y in np.linspace(0.2, 0.8, 40):
                points.append((x, y, layout.table_top))
                colors.append((140, 102, 64))
    return np.array(colors, dtype=np.uint8)[None], np.array(points, dtype=float)[None]


def test_locate_bins_finds_moved_bins_and_ignores_the_table():
    layout = Layout()
    centres = {"good": (0.515, 0.365), "rework": (0.425, 0.588)}
    found = locate_bins(*rim_cloud(layout, centres), layout)
    assert set(found) == {"good", "rework"}
    for name, (cx, cy) in centres.items():
        assert abs(found[name][0] - cx) < 0.004 and abs(found[name][1] - cy) < 0.004


def test_a_half_hidden_bin_is_reported_missing_not_guessed():
    layout = Layout()
    rgb, xyz = rim_cloud(layout, {"good": (0.50, 0.38), "rework": (0.44, 0.57)}, table=False)
    keep = ~((xyz[..., 1] > 0.47) & (xyz[..., 0] > 0.44))  # something covers the far half of the rework tray
    found = locate_bins(rgb[keep][None], xyz[keep][None], layout)
    assert "good" in found and "rework" not in found


def test_bin_jitter_moves_bins_without_changing_a_seeds_parts():
    still, moving = Layout(), Layout(bin_jitter_m=0.02)
    a, b = make_episode(still, 5), make_episode(moving, 5)
    assert a.parts == b.parts and a.poses == b.poses
    assert a.bin_xy == still.bin_xy
    assert b.bin_xy != still.bin_xy
    for (_, x, y), (_, x0, y0) in zip(b.bin_xy, still.bin_xy, strict=True):
        assert abs(x - x0) <= 0.02 and abs(y - y0) <= 0.02


def test_sort_aims_at_the_bins_it_saw_not_the_drawing(tmp_path: Path):
    layout = Layout(bin_jitter_m=0.02)
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, 1)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    holder = {}
    ctx = SortContext(layout=layout, episode=episode, ladder=Ladder([Step(OracleRung(lambda q: truth(sim, holder["ctx"], holder["part"], q.id)), 0.0)]), logbook=Logbook(tmp_path, "ep1"))
    holder["ctx"] = ctx
    original = sort_module.ask

    def ask(sim_, ctx_, qid, part):
        holder["part"] = part
        return original(sim_, ctx_, qid, part)

    sort_module.ask = ask
    try:
        result = run_sort(sim, ctx)
    finally:
        sort_module.ask = original
    true_xy = {name: (x, y) for name, x, y in episode_layout(layout, episode).bin_xy}
    assert set(ctx.bins_seen) == set(true_xy)
    assert sim.bins_seen is ctx.bins_seen  # the bump guard's box estimate uses the trays where they were seen, outside the demo too
    for name, (x, y) in ctx.bins_seen.items():
        assert abs(x - true_xy[name][0]) < 0.005 and abs(y - true_xy[name][1]) < 0.005
        nominal = {n: (cx, cy) for n, cx, cy in layout.bin_xy}[name]
        assert (x, y) != nominal
    assert result.checks["parts_in_correct_bin"] == result.checks["n_parts"], result.trace.text()
