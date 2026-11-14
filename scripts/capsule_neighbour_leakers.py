"""Which leak_block "yes" pictures owe their yes to a neighbour block's leaker, and is that leaker's pool all in view (Nov 5).

Review round 7 found leak_block pictures labelled yes only because a leaker from the next block lies wholly inside
the 15 mm pad. A leaker is told apart only by its pool (capsule_episode: nothing else differs), and a pool is a disc
up to 30 mm across centred on the capsule, so a capsule wholly in the picture can still have its pool cut by the
edge. This sorts every yes picture by why it is yes and whether the pool is whole, then counts how each replayed arm
answered each kind, so the label rule for those pictures is chosen on what the pictures show.

    python3 scripts/capsule_neighbour_leakers.py output/capsules/sel01
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

# capsule_cell.POOL_SIZES (pool diameters in metres), copied so this runs on plain Python without the sim stack
POOL_SIZES = (0.016, 0.022, 0.030)


def wholly_in(box, window) -> bool:
    top, left, bottom, right = window
    return box[0] >= top and box[1] >= left and box[2] <= bottom and box[3] <= right


def pool_box(cap: dict, px_per_m: float) -> tuple[float, float, float, float]:
    r = (cap["box"][0] + cap["box"][2]) / 2
    c = (cap["box"][1] + cap["box"][3]) / 2
    half = POOL_SIZES[cap["pool_size"]] * px_per_m / 2
    return (r - half, c - half, r + half, c + half)


def kinds(log: Path, tally: collections.Counter | None = None) -> dict[int, str]:
    """index -> own / neighbour_pool_whole / neighbour_pool_cut, for every leak_block row labelled yes.

    ``tally`` (if given) also counts the yes rows holding any neighbour leaker wholly (also those with an own leaker
    too) and the pool size index of each neighbour leaker in the neighbour-only rows, by whether its pool is whole."""
    out: dict[int, str] = {}
    tally = tally if tally is not None else collections.Counter()
    tray = None
    for i, line in enumerate((log / "decisions.jsonl").open(encoding="utf-8")):
        r = json.loads(line)
        ctx = r.get("context") or {}
        if r["question"] != "leak_block":
            continue
        if ctx.get("tray"):
            tray = ctx["tray"]
        if r.get("truth") != "yes" or tray is None:
            continue
        caps, window, own = tray["capsules"], tuple(ctx["window"]), set(ctx["capsules"])
        whole = [k for k in ctx["in_view"] if caps[k]["leaker"] and wholly_in(caps[k]["box"], window)]
        neighbours = [k for k in whole if k not in own]
        if neighbours:
            tally["with a neighbour leaker wholly in view"] += 1
            if len(neighbours) < len(whole):
                tally["  of them also holding an own leaker"] += 1
            else:
                for k in neighbours:
                    pool_whole = wholly_in(pool_box(caps[k], tray["px_per_m"]), window)
                    tally[f"  neighbour-only leaker, pool size {caps[k]['pool_size']}, pool {'whole' if pool_whole else 'cut'}"] += 1
        if any(k in own for k in whole):
            out[i] = "own"
        elif any(wholly_in(pool_box(caps[k], tray["px_per_m"]), window) for k in whole):
            out[i] = "neighbour_pool_whole"
        else:
            out[i] = "neighbour_pool_cut"
    return out


def main() -> int:
    for arg in sys.argv[1:]:
        log = Path(arg)
        tally: collections.Counter = collections.Counter()
        kind = kinds(log, tally)
        print("==", log, dict(collections.Counter(kind.values())))
        for key, n in tally.items():  # first-seen order: the "with a neighbour" line always comes before its sub-lines
            print(f"  {key}: {n}")
        by: dict = collections.defaultdict(collections.Counter)
        for line in (log / "arms.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            k = kind.get(r["index_global"])
            if k is None or r.get("error"):
                continue
            p = r["probs"]
            top = max(p, key=p.get)
            c = by[(r["arm"], k)]
            c["n"] += 1
            c["yes"] += top == "yes"
            c["no_at_0.9"] += top == "no" and p[top] >= 0.9
        for (arm, k), c in sorted(by.items()):
            print(f"  {arm:22s} {k:22s} n {c['n']:4d}  said yes {c['yes'] / c['n']:.2f}  said no at >=0.9 {c['no_at_0.9']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
