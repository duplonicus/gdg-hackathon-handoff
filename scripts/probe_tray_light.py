"""How evenly is the tray lit, and what casts the shade bands? (Nov 6; an experiment beside the setup, not a change)

sel02's clean single-capsule pictures in shade drew 62% false alarms from d3 against 15% lit (capsule_shade_check.py).
This builds the two-cell scene under a few lighting variants, parks the robot at cell a's station in the travel pose
(as the scan sees it), takes the tray camera's picture, and measures the bare parchment's brightness: its spread and
its darkest 5%. Pictures go beside the numbers. Nothing here changes how the task runs.

    .\\run.ps1 scripts/probe_tray_light.py --variant 0 --out output/cells/tray_light   (then 1, 2, 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import LAMP_HEIGHT, LAMP_INTENSITY, TwoCells, make_two_cell_scene  # noqa: E402
from handoff.capsule_episode import make_capsule_episode, place_capsule_episode  # noqa: E402
from handoff.capsule_skills import park_station, travel_pose  # noqa: E402


def variants(scene, cells):
    lay = cells.layout
    a, b = cells.cells
    corners = []
    for cell in (a, b):
        cx, cy = lay.tray_center
        for dy in (-0.25, 0.25):  # two a tray: eight lamps and the sun passed Genesis's light limit (Nov 6)
            corners.append((*cell.to_world(cx, cy + dy), lay.table_top + LAMP_HEIGHT, LAMP_INTENSITY / 2.0))
    return {
        "now (sun 5, one lamp, ambient 0.3)": scene,
        "no sun": replace(scene, sun=0.0),
        "no sun, 2 lamps a tray": replace(scene, sun=0.0, lamps=tuple(corners)),
        "no sun, one lamp, ambient 0.6": replace(scene, sun=0.0, ambient=0.6),
    }


def paper_stats(rgb: np.ndarray) -> dict:
    """Bare parchment = bright, low-saturation pixels in the middle 80% of the picture (capsules are amber, saturated)."""
    h, w, _ = rgb.shape
    mid = rgb[h // 10 : -h // 10, w // 10 : -w // 10].astype(float)
    lum = mid.mean(axis=2)
    sat = mid.max(axis=2) - mid.min(axis=2)
    paper = lum[(sat < 40) & (lum > 120)]
    return {"pixels": int(paper.size), "median": round(float(np.median(paper)), 1), "p05": round(float(np.percentile(paper, 5)), 1),
            "spread_p95_p05": round(float(np.percentile(paper, 95) - np.percentile(paper, 5)), 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--config", default=str(ROOT / "config" / "capsules.yaml"))
    parser.add_argument("--variant", type=int, required=True, help="0 to 3 (one Genesis scene per process)")
    parser.add_argument("--out", default=str(ROOT / "output" / "cells" / "tray_light"))
    args = parser.parse_args()

    from PIL import Image

    from handoff.sim.genesis_sim import GenesisSimulator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = TwoCells()
    results = {}
    for k, (label, scene) in enumerate(variants(make_two_cell_scene(cells), cells).items()):
        if k != args.variant:
            continue
        sim = GenesisSimulator(args.config, sensors=None, help_text=False, line_step_m=0.05)
        sim.build(scene)
        episode = make_capsule_episode(cells, args.seed)
        place_capsule_episode(sim, cells, episode)
        work = episode.work_cell or "a"  # photograph the cell with capsules on its tray
        park_station(sim, cells.cell(work))
        travel_pose(sim)
        rgb = sim.capture(f"{work}_tray").rgb
        Image.fromarray(rgb).save(out / f"v{k}.png")
        results[label] = paper_stats(rgb)
        print(json.dumps({label: results[label]}), flush=True)
    with (out / "light.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
