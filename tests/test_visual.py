"""Tests for load-time decimation of visual meshes."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mh3_genesis.visual import decimate_trimesh

NO_CACHE = {"cache_dir": None}


def _dense_sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=5)  # 20480 faces


def test_decimate_keeps_about_the_ratio() -> None:
    mesh = _dense_sphere()
    reduced = decimate_trimesh(mesh, 0.1, **NO_CACHE)
    assert reduced is not None
    assert len(reduced.faces) <= 2048
    assert len(reduced.faces) >= 1500
    # Still a sphere of about the same size.
    assert np.allclose(reduced.bounds, mesh.bounds, atol=0.05)


def test_small_mesh_is_left_alone() -> None:
    assert decimate_trimesh(trimesh.creation.box(), 0.1, **NO_CACHE) is None


def test_triangle_soup_is_welded_not_torn() -> None:
    # Unweld a sphere into one triangle per face (what the MH3 meshes look like) and check the result is a
    # closed surface of about the right size, not a pile of deleted triangles.
    mesh = _dense_sphere()
    soup = trimesh.Trimesh(mesh.vertices[mesh.faces].reshape(-1, 3), np.arange(len(mesh.faces) * 3).reshape(-1, 3), process=False)
    assert len(soup.vertices) == 3 * len(soup.faces)
    reduced = decimate_trimesh(soup, 0.1, **NO_CACHE)
    assert reduced is not None
    assert reduced.is_watertight
    assert np.allclose(reduced.bounds, mesh.bounds, atol=0.05)


def test_textured_mesh_is_left_alone() -> None:
    mesh = _dense_sphere()
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    assert decimate_trimesh(mesh, 0.1, **NO_CACHE) is None


def test_cache_round_trip(tmp_path) -> None:
    mesh = _dense_sphere()
    first = decimate_trimesh(mesh, 0.1, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    again = decimate_trimesh(mesh, 0.1, cache_dir=tmp_path)
    assert first is not None and again is not None
    assert np.array_equal(first.faces, again.faces)
    # A different ratio is a different cache entry.
    decimate_trimesh(mesh, 0.2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_bad_ratio_is_rejected_by_config() -> None:
    from dataclasses import replace

    from mh3_genesis.config import SimulatorConfig

    base = SimulatorConfig.from_yaml("config/mh3.yaml")
    assert base.visual_face_ratio is None
    assert base.viewer_refresh_rate == 60
    replace(base, visual_face_ratio=0.25, viewer_refresh_rate=30).validate()
    with pytest.raises(ValueError, match="visual_face_ratio"):
        replace(base, visual_face_ratio=0.0).validate()
    with pytest.raises(ValueError, match="visual_face_ratio"):
        replace(base, visual_face_ratio=1.5).validate()
    with pytest.raises(ValueError, match="viewer_refresh_rate"):
        replace(base, viewer_refresh_rate=0).validate()


def test_demo_config_draws_the_full_robot_at_60_hz() -> None:
    """The demo runs on native Windows, where the full meshes hold real time; 0.25 and 30 Hz are the WSL values."""
    from mh3_genesis.config import SimulatorConfig

    demo = SimulatorConfig.from_yaml("config/handoff_demo.yaml")
    assert demo.visual_face_ratio is None
    assert demo.viewer_refresh_rate == 60
