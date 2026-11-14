"""Load-time decimation of visual meshes.

The MH3 USD carries about 2.3 million vertices of visual geometry. Genesis draws every visual mesh on every
viewer frame and camera render, and under WSL that goes through Mesa (llvmpipe or the D3D12 layer), which
processes vertices on the CPU: about 120 ms a frame for the full robot, 3 ms for a light scene. Measured
Oct 11, 2026. Welding and decimating the visual meshes to a fraction of their faces makes the viewer run close to real
time again; physics and collision meshes are not involved. On native Windows with the NVIDIA driver the full
robot already draws at real time (measured the same day), so this is for WSL and other software-GL setups. Only the robot is decimated (the stage load is
wrapped), so the parts the models inspect keep their exact meshes and textures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


MIN_FACES = 500
CACHE_DIR = Path.home() / ".cache" / "mh3_genesis" / "visual"


def decimate_trimesh(mesh: Any, ratio: float, min_faces: int = MIN_FACES, cache_dir: Path | None = CACHE_DIR) -> Any | None:
    """Return a welded copy of a trimesh keeping about ``ratio`` of its faces, or None when it should stay as is.

    The target is ``ratio`` times the face count, never below ``min_faces``, so large curved panels keep
    thousands of faces while small brackets keep what they have. A mesh stays as is when it carries a UV map:
    decimation does not carry UVs, so a textured mesh would lose its texture mapping.

    Uses pymeshlab's quadric edge collapse with normal, topology and boundary preservation. The MH3 shells are
    CAD exports with inconsistent winding, and the cheaper ``fast_simplification`` tears and creases them
    (tried Oct 11, 2026); pymeshlab keeps them clean at a quarter of the faces but takes about 10 s for the
    whole robot, so results are cached on disk by mesh content and ratio.

    :param mesh: A ``trimesh.Trimesh``.
    :type mesh: trimesh.Trimesh
    :param ratio: Fraction of faces to keep, in (0, 1].
    :type ratio: float
    :param min_faces: Floor for the target face count.
    :type min_faces: int
    :param cache_dir: Where decimated meshes are cached, or None for no cache.
    :type cache_dir: Path | None
    :return: The reduced mesh, or None.
    :rtype: trimesh.Trimesh | None
    """
    import trimesh

    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.uv is not None:
        return None
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    target = max(min_faces, int(len(faces) * ratio))
    if len(faces) <= target:
        return None

    key = None
    if cache_dir is not None:
        digest = hashlib.sha1(vertices.tobytes() + faces.tobytes() + f"{ratio:.4f}/{min_faces}".encode()).hexdigest()
        key = cache_dir / f"{digest}.npz"
        if key.is_file():
            cached = np.load(key)
            return trimesh.Trimesh(cached["vertices"], cached["faces"], process=False)

    import pymeshlab

    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    # Weld first: the MH3 visual meshes are triangle soup (every face has its own three vertices), and edge
    # collapse on unconnected triangles just deletes them, leaving holes.
    meshes.meshing_remove_duplicate_vertices()
    if meshes.current_mesh().face_number() > target:
        meshes.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target,
            preservenormal=True,
            preservetopology=True,
            preserveboundary=True,
            planarquadric=True,
            qualitythr=0.3,
        )
    result = meshes.current_mesh()
    out_vertices = np.ascontiguousarray(result.vertex_matrix(), dtype=np.float64)
    out_faces = np.ascontiguousarray(result.face_matrix(), dtype=np.int64)
    if key is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(key, vertices=out_vertices, faces=out_faces)
    return trimesh.Trimesh(out_vertices, out_faces, process=False)


@contextmanager
def decimated_visual_meshes(gs: Any, ratio: float | None) -> Iterator[None]:
    """While active, every visual mesh Genesis attaches to a rigid link keeps about ``ratio`` of its faces.

    Patches ``RigidLink._add_vgeom`` for the duration, so wrap only the entity loads that should be lighter
    (the robot), not the whole scene. A None ``ratio`` does nothing.

    :param gs: The imported ``genesis`` module.
    :type gs: module
    :param ratio: Fraction of faces to keep per visual mesh, or None to leave meshes alone.
    :type ratio: float | None
    """
    if ratio is None:
        yield
        return
    from genesis.engine.entities.rigid_entity import rigid_link

    original = rigid_link.RigidLink._add_vgeom
    stats = {"meshes": 0, "decimated": 0, "before": 0, "after": 0, "seconds": 0.0}

    def _add_vgeom(self: Any, desc: Any) -> Any:
        vmesh = desc.vmesh
        source = vmesh.trimesh
        stats["meshes"] += 1
        stats["before"] += len(source.vertices)
        started = time.perf_counter()
        reduced = decimate_trimesh(source, ratio)
        if reduced is not None:
            # Keep the surface (colour or texture settings) and metadata; only the geometry changes.
            lighter = gs.Mesh.from_trimesh(reduced, surface=vmesh.surface, metadata=vmesh.metadata)
            desc = dataclasses.replace(desc, vmesh=lighter)
            stats["decimated"] += 1
        stats["seconds"] += time.perf_counter() - started
        stats["after"] += len(desc.vmesh.trimesh.vertices)
        return original(self, desc)

    rigid_link.RigidLink._add_vgeom = _add_vgeom
    try:
        yield
    finally:
        rigid_link.RigidLink._add_vgeom = original
    LOGGER.info(
        "visual meshes: %d of %d decimated to %.0f%% of their faces, %d -> %d vertices in %.1f s",
        stats["decimated"], stats["meshes"], 100 * ratio, stats["before"], stats["after"], stats["seconds"],
    )
    print(
        f"visual meshes: {stats['decimated']} of {stats['meshes']} decimated to {100 * ratio:.0f}% of their faces, "
        f"{stats['before']:,} -> {stats['after']:,} vertices in {stats['seconds']:.1f} s"
    )
