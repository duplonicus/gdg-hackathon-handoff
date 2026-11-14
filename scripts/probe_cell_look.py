"""The capsule cells' new floor and table, against the old look, in every camera the models see (Nov 6; a check, not a run).

dup, Nov 6: a whitish grey tile floor in place of the kit's blue checker, and a steel table, with no shine if shine
would touch speed or results. This builds the two-cell scene one way per process (``--look new`` or ``--look old``: the
kit's checker plane and the brown table), places one episode, parks the robot at its station in the travel pose (as the
scan sees it) and saves every scene camera's picture plus a wide shot. ``--compare`` then reports, per camera, how far
the new pictures are from the old ones and the tray's bare-parchment brightness (the lighting probe's measure), so a
change in what the models see shows as a number.

    .\\run.ps1 scripts/probe_cell_look.py --look old --out output/cells/look
    .\\run.ps1 scripts/probe_cell_look.py --look new --out output/cells/look
    .\\run.ps1 scripts/probe_cell_look.py --compare --out output/cells/look
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TABLE_COLOR, TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode  # noqa: E402
from handoff.capsule_skills import park_station, travel_pose  # noqa: E402
from handoff.sim.base import CameraSpec  # noqa: E402

OLD_TABLE = (0.55, 0.40, 0.25)  # the table's colour before Nov 6


def old_look(scene):
    boxes = []
    for box in scene.boxes:
        if box.name == "floor_tiles":
            continue
        if box.color == TABLE_COLOR and "table" in box.name:
            box = replace(box, color=OLD_TABLE, look="plastic")
        boxes.append(box)
    return replace(scene, boxes=tuple(boxes), kit_floor=True)


def wide(cells: TwoCells) -> CameraSpec:
    a, b = cells.cells
    mid = ((a.origin[0] + b.origin[0]) / 2.0, (a.origin[1] + b.origin[1]) / 2.0)
    return CameraSpec("look_wide", pos=(mid[0] + 2.6, mid[1] - 2.6, 2.2), lookat=(*mid, 0.6), fov=50, res=(1280, 800))


def shoot(look: str, seed: int, config: str, out: Path) -> None:
    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    cells = TwoCells()
    scene = make_two_cell_scene(cells)
    if look == "old":
        scene = old_look(scene)
    scene = replace(scene, cameras=(*scene.cameras, wide(cells)))
    sim = GenesisSimulator(config, sensors=None, help_text=False, line_step_m=0.05)
    t0 = time.perf_counter()
    sim.build(scene)
    build_s = time.perf_counter() - t0
    episode = make_capsule_episode(cells, seed)
    place_capsule_episode(sim, cells, episode)
    park_station(sim, cells.cell(episode.work_cell or "a"))
    travel_pose(sim)
    folder = out / look
    folder.mkdir(parents=True, exist_ok=True)
    timings = {}
    for cam in scene.cameras:
        t0 = time.perf_counter()
        rgb = sim.capture(cam.name).rgb
        timings[cam.name] = round(time.perf_counter() - t0, 3)
        Image.fromarray(rgb).save(folder / f"{cam.name}.png")
    print(json.dumps({"look": look, "seed": seed, "build_s": round(build_s, 1), "capture_s": timings}), flush=True)


def paper_stats(rgb: np.ndarray) -> dict:
    """As probe_tray_light.py: bright, low-saturation pixels in the middle 80% of the tray picture. With the steel table,
    grey table pixels pass this filter too (a growing ``pixels`` count shows it); PARCHMENT is the real comparison."""
    h, w, _ = rgb.shape
    mid = rgb[h // 10 : -h // 10, w // 10 : -w // 10].astype(float)
    lum = mid.mean(axis=2)
    sat = mid.max(axis=2) - mid.min(axis=2)
    paper = lum[(sat < 40) & (lum > 120)]
    return {"pixels": int(paper.size), "median": round(float(np.median(paper)), 1), "p05": round(float(np.percentile(paper, 5)), 1),
            "spread_p95_p05": round(float(np.percentile(paper, 95) - np.percentile(paper, 5)), 1)}


PARCHMENT = (0.25, 0.817, 0.234, 0.771)  # rows then columns, as fractions: inside the tray rim, parchment only


def compare(out: Path) -> None:
    from PIL import Image

    for path in sorted((out / "new").glob("*.png")):
        old_path = out / "old" / path.name
        if not old_path.exists():
            continue
        new, old = (np.asarray(Image.open(p).convert("RGB")).astype(float) for p in (path, old_path))
        diff = np.abs(new - old).max(axis=2)
        row = {"camera": path.stem, "mean_abs_diff": round(float(diff.mean()), 2), "pixels_changed_pct": round(float((diff > 8).mean() * 100), 2)}
        if path.stem.endswith("_tray"):
            row["paper_old"], row["paper_new"] = paper_stats(old.astype(np.uint8)), paper_stats(new.astype(np.uint8))
            h, w = diff.shape
            r0, r1, c0, c1 = PARCHMENT
            inside = diff[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w)]
            row["parchment_max_diff"], row["parchment_pct_over_3"] = int(inside.max()), round(float((inside > 3).mean() * 100), 2)
        print(json.dumps(row), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--look", choices=("new", "old"))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "look"))
    args = parser.parse_args()
    out = Path(args.out)
    if args.compare:
        compare(out)
    elif args.look:
        shoot(args.look, args.seed, args.config, out)
    else:
        parser.error("--look new|old or --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
