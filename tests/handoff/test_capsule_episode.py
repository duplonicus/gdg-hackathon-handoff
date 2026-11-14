"""Capsule episodes (Nov 4): deterministic, believable, and placed without anything clipping."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from handoff.capsule_cell import MAX_LEAKERS, POOL_SIZES, TwoCells, make_two_cell_scene, overlapping, visible
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode
from handoff.sim.fake import FakeSimulator


def test_an_episode_is_the_same_for_the_same_seed():
    cells = TwoCells()
    assert make_capsule_episode(cells, 7) == make_capsule_episode(cells, 7)
    assert make_capsule_episode(cells, 7) != make_capsule_episode(cells, 8)


def test_the_mix_over_many_seeds():
    cells = TwoCells()
    episodes = [make_capsule_episode(cells, s) for s in range(400)]
    where = Counter(e.work_cell for e in episodes)
    assert 0.35 < where["a"] / 400 < 0.55 and 0.35 < where["b"] / 400 < 0.55 and 0.04 < where[None] / 400 < 0.16
    for e in episodes:
        assert len(e.leakers) == len(set(e.leakers)) == len(e.pool_sizes) <= MAX_LEAKERS
        assert all(0 <= s < len(POOL_SIZES) for s in e.pool_sizes)
        if e.work_cell is None:
            assert e.leakers == ()
    assert any(len(e.leakers) == 0 and e.work_cell for e in episodes)  # a clean tray happens too


def scene_as_placed(sim, spec):
    """The scene's boxes where the fake simulator has them now."""
    return [replace(b, pos=tuple(float(v) for v in sim.object_state(b.name).pos)) for b in spec.boxes]


@pytest.mark.parametrize("seed", range(12))
def test_placing_an_episode(seed):
    cells = TwoCells()
    spec = make_two_cell_scene(cells)
    sim = FakeSimulator()
    sim.build(spec)
    ep = make_capsule_episode(cells, seed)
    place_capsule_episode(sim, cells, ep)
    lay = cells.layout
    placed = scene_as_placed(sim, spec)
    assert overlapping(visible(placed)) == []
    on_paper = [b for b in placed if b.name.startswith("capsule_") and b.pos[2] > 0]
    if ep.work_cell is None:
        assert on_paper == []
        return
    cell = cells.cell(ep.work_cell)
    assert len(on_paper) == len(lay.grid())  # leakers are tray capsules like the rest: only the pool tells them apart
    assert all(sim.object_state(n).pos[0] > 2.5 for n in (*cells.leaker_names, *cells.spare_names))  # loose bodies wait offstage
    for i, (gx, gy) in enumerate(lay.grid()):
        pos = np.array(sim.object_state(cells.grid_names[i]).pos[:2])
        assert np.hypot(*(pos - np.array(cell.to_world(gx, gy)))) <= 0.0012  # nudged, never more than the jitter
    for k, spot in enumerate(ep.leakers):
        under = np.array(sim.object_state(cells.grid_names[spot]).pos[:2])
        pool = np.array(sim.object_state(cells.pool_name(k, ep.pool_sizes[k])).pos)
        assert np.allclose(pool[:2], under) and pool[2] > lay.paper_top  # the pool sits under its capsule
    for each in cells.cells:
        shown = "work" if each.name == ep.work_cell else "done"
        hidden = "done" if shown == "work" else "work"
        assert all(b.pos[2] - b.size[2] / 2.0 >= 0.0 for b in placed if b.name.startswith(f"{each.name}_{shown}_"))  # in view
        assert all(b.pos[2] + b.size[2] / 2.0 < 0.0 for b in placed if b.name.startswith(f"{each.name}_{hidden}_"))  # under the floor


def test_placing_twice_in_a_row_leaves_nothing_behind():
    cells = TwoCells()
    spec = make_two_cell_scene(cells)
    sim = FakeSimulator()
    sim.build(spec)
    first = next(make_capsule_episode(cells, s) for s in range(50) if make_capsule_episode(cells, s).work_cell == "a" and len(make_capsule_episode(cells, s).leakers) >= 3)
    second = next(make_capsule_episode(cells, s) for s in range(50) if make_capsule_episode(cells, s).work_cell == "b" and len(make_capsule_episode(cells, s).leakers) == 1)
    place_capsule_episode(sim, cells, first)
    place_capsule_episode(sim, cells, second)
    shown_pools = [n for n in cells.all_pool_names if sim.object_state(n).pos[2] > 0]
    assert shown_pools == [cells.pool_name(0, second.pool_sizes[0])]
    offstage = [n for n in cells.leaker_names if sim.object_state(n).pos[0] > 2.5]
    assert len(offstage) == MAX_LEAKERS


def test_the_capsules_are_not_a_perfect_grid_and_the_same_seed_nudges_them_the_same_way():
    from handoff.capsule_episode import capsule_poses

    cells = TwoCells()
    ep = make_capsule_episode(cells, 3)
    poses = capsule_poses(cells, ep)
    assert poses == capsule_poses(cells, ep)
    yaws = np.array([p[2] for p in poses]) - cells.cell(ep.work_cell).yaw_deg
    assert yaws.std() > 2.0 and np.abs(yaws).max() <= 8.0
