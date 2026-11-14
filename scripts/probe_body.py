"""How big is the MH3 around its base, link by link? (Nov 4: can its base tuck under a table top?)

Prints every link's collision bounding box relative to the base origin (x ahead, y left, z up from the floor), at
the kit's home pose, and writes them to JSON.

    .\\run.ps1 scripts/probe_body.py --out output/bench/body.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.sim.base import SceneSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "config" / "handoff.yaml"))
    parser.add_argument("--out", default=str(ROOT / "output" / "bench" / "body.json"))
    args = parser.parse_args()

    from handoff.sim.genesis_sim import GenesisSimulator

    sim = GenesisSimulator(args.config)
    sim.build(SceneSpec())
    sim.settle(0.5)
    base = np.asarray(sim.base_pose().pos, dtype=float)
    kit = sim.kit
    rows = []
    for name in kit.link_names:
        link = kit.get_link(name)
        try:
            lo, hi = (np.asarray(v.cpu() if hasattr(v, 'cpu') else v, dtype=float).reshape(-1)[:3] for v in link.get_AABB())
        except Exception as error:  # a link with no collision geometry
            rows.append({"link": name, "error": str(error)[:80]})
            continue
        rows.append({"link": name, "min": np.round(lo - [base[0], base[1], 0.0], 3).tolist(), "max": np.round(hi - [base[0], base[1], 0.0], 3).tolist()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"base": base.tolist(), "links": rows}, indent=1), encoding="utf-8")
    for r in rows:
        if "min" in r:
            print(f"{r['link']:28s} x {r['min'][0]:+.2f}..{r['max'][0]:+.2f}  y {r['min'][1]:+.2f}..{r['max'][1]:+.2f}  z {r['min'][2]:.2f}..{r['max'][2]:.2f}")
        else:
            print(f"{r['link']:28s} {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
