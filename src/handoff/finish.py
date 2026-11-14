"""Part finishes: good, scratched, unpainted, with a severity dial, drawn as textures.

The sim knows each part's true finish; the models only see the camera. Severity runs from 0
(faint: a couple of thin, low-contrast scratches; a small bare patch) to 1 (obvious: many bright
scratches; the whole part bare). It is the difficulty dial that makes the ladder escalate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINISHES = ("good", "scratched", "unpainted")
PAINT = np.array([45, 90, 200], dtype=float)  # blue paint
BARE = np.array([165, 165, 170], dtype=float)  # bare metal
SCRATCH = np.array([215, 215, 225], dtype=float)
ASSETS = Path(__file__).resolve().parent / "assets"
CUBE_OBJ = ASSETS / "cube_uv.obj"
GENERATED = ASSETS / "generated"  # meshes made at run time (gitignored)


@dataclass(frozen=True)
class Finish:
    kind: str  # one of FINISHES
    severity: float = 0.0  # 0 faint .. 1 obvious; 0 for "good"

    @property
    def defective(self) -> bool:
        return self.kind != "good"

    @property
    def label(self) -> str:
        return f"{self.kind}@{self.severity:.2f}"


def finish_pool(per_kind: int, seed: int = 0) -> list[Finish]:
    """``per_kind`` parts of each finish, defect severities spread evenly from faint to obvious."""
    rng = np.random.default_rng(seed)
    pool: list[Finish] = [Finish("good", 0.0) for _ in range(per_kind)]
    for kind in ("scratched", "unpainted"):
        levels = np.linspace(0.1, 1.0, per_kind) if per_kind > 1 else np.array([0.6])
        rng.shuffle(levels)
        pool.extend(Finish(kind, float(round(s, 2))) for s in levels)
    return pool


def texture(finish: Finish, seed: int = 0, size: int = 128) -> np.ndarray:
    """An RGB uint8 image of one face of the part (the same image wraps every face)."""
    rng = np.random.default_rng(seed)
    img = np.broadcast_to(PAINT, (size, size, 3)).astype(float).copy()
    img += rng.normal(0.0, 4.0, img.shape)  # paint grain
    if finish.kind == "scratched":
        n_lines = 1 + int(round(5 * finish.severity))
        alpha = 0.25 + 0.75 * finish.severity
        width = 1 if finish.severity < 0.5 else 2
        for _ in range(n_lines):
            x0, y0 = rng.uniform(size * 0.1, size * 0.9, size=2)
            angle = rng.uniform(0, np.pi)
            length = rng.uniform(size * 0.3, size * 0.8)
            for t in np.arange(0.0, length, 0.5):
                x, y = int(x0 + t * np.cos(angle)), int(y0 + t * np.sin(angle))
                for dx in range(width):
                    for dy in range(width):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < size and 0 <= yy < size:
                            img[yy, xx] = img[yy, xx] * (1 - alpha) + SCRATCH * alpha
    elif finish.kind == "unpainted":
        fraction = 0.15 + 0.85 * finish.severity
        if fraction >= 0.99:
            img[:] = BARE + rng.normal(0.0, 4.0, img.shape)
        else:
            # one bare rectangle covering `fraction` of the face, placed at random
            area = fraction * size * size
            w = int(np.clip(np.sqrt(area) * rng.uniform(0.7, 1.4), 4, size))
            h = int(np.clip(area / w, 4, size))
            x0 = int(rng.integers(0, size - w + 1))
            y0 = int(rng.integers(0, size - h + 1))
            img[y0 : y0 + h, x0 : x0 + w] = BARE + rng.normal(0.0, 4.0, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def write_cube_obj(path: Path = CUBE_OBJ) -> Path:
    """A unit cube whose six faces each map to the whole texture (Genesis boxes have no UVs)."""
    faces = (
        ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 0, -1), [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
        ((1, 0, 0), [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
        ((-1, 0, 0), [(-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 1, 0), [(0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]),
        ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
    )
    uv = ((0, 0), (1, 0), (1, 1), (0, 1))
    lines = ["# unit cube, each face mapped to the full texture"]
    for normal, corners in faces:
        lines.extend(f"v {x} {y} {z}" for x, y, z in corners)
        lines.extend(f"vt {u} {v}" for u, v in uv)
        lines.append(f"vn {normal[0]} {normal[1]} {normal[2]}")
    for i in range(6):
        b, n = 4 * i + 1, i + 1
        lines.append(f"f {b}/{b}/{n} {b + 1}/{b + 1}/{n} {b + 2}/{b + 2}/{n}")
        lines.append(f"f {b}/{b}/{n} {b + 2}/{b + 2}/{n} {b + 3}/{b + 3}/{n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def capsule_obj(length: float, diameter: float, height: float | None = None, folder: Path = GENERATED) -> Path:
    """A pill mesh lying along x: a cylinder with two rounded ends, ``length`` end to end, ``diameter`` across (y).

    ``height`` (z) below ``diameter`` squashes the cross-section to an oval. Round pills settle within 2 mm on the
    parchment (capsule_count_probe, Nov 4), so the cells use round ones."""
    height = diameter if height is None else height
    if not diameter < length or not 0 < height <= diameter:
        raise ValueError(f"a capsule needs 0 < height <= diameter < length, got {height}, {diameter}, {length}")
    path = folder / f"capsule_{length * 1000:.1f}x{diameter * 1000:.1f}x{height * 1000:.1f}mm.obj"
    if not path.exists():
        import trimesh

        mesh = trimesh.creation.capsule(height=length - diameter, radius=diameter / 2.0, count=[24, 12])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
        mesh.apply_scale([1.0, 1.0, height / diameter])
        folder.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path))
    return path
