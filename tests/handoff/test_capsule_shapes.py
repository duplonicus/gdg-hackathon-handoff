"""Capsule meshes and drawn-only shapes (Nov 4): the pill's real size, and pools that physics ignores."""

from __future__ import annotations

import numpy as np
import pytest

from handoff.finish import capsule_obj
from handoff.sim.base import BoxSpec, SceneSpec
from handoff.sim.fake import FakeSimulator


@pytest.mark.parametrize("size", [(0.024, 0.011, 0.011), (0.024, 0.011, 0.0085), (0.08, 0.04, 0.03)])
def test_capsule_mesh_bounds_match_its_size(tmp_path, size):
    import trimesh

    mesh = trimesh.load(capsule_obj(*size, folder=tmp_path), force="mesh")
    extent = mesh.bounds[1] - mesh.bounds[0]
    np.testing.assert_allclose(extent, size, atol=2e-4)  # lies along x, width y, height z
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0.0, atol=2e-4)  # centred, so pos is the pill's centre
    assert mesh.is_watertight


def test_capsule_mesh_is_made_once_per_size(tmp_path):
    first = capsule_obj(0.024, 0.011, 0.0085, folder=tmp_path)
    stamp = first.stat().st_mtime_ns
    assert capsule_obj(0.024, 0.011, 0.0085, folder=tmp_path) == first
    assert first.stat().st_mtime_ns == stamp
    assert capsule_obj(0.024, 0.011, 0.011, folder=tmp_path) != first


@pytest.mark.parametrize("size", [(0.01, 0.011, 0.011), (0.024, 0.011, 0.012), (0.024, 0.011, 0.0)])
def test_capsule_mesh_rejects_impossible_sizes(tmp_path, size):
    with pytest.raises(ValueError):
        capsule_obj(*size, folder=tmp_path)


def test_fake_sim_does_not_stand_parts_on_a_drawn_only_pool():
    table = BoxSpec("table", (1.0, 1.0, 0.8), (0.5, 0.0, 0.4), fixed=True)
    pool = BoxSpec("pool", (0.03, 0.03, 0.002), (0.5, 0.0, 0.801), fixed=True, shape="disc", collision=False)
    pill = BoxSpec("pill", (0.024, 0.011, 0.0085), (0.5, 0.0, 0.9), shape="capsule")
    sim = FakeSimulator()
    sim.build(SceneSpec(boxes=(table, pool, pill)))
    support, top = sim._support("pill")
    assert support == "table"
    assert top == pytest.approx(0.8)
