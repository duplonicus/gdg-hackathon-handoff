"""Cut the logged tray pictures into blocks of any size, labelled, for "how many capsules per question" (Nov 4).

dup asked how far the leaker check can be pushed: a person spots a leaker on a tray of hundreds at a glance. A capsule
log keeps every scan's whole-tray picture with each capsule's pixel box and the simulator's truth (leaker or not,
and its pool size), so this cuts the same pictures into blocks of 1, 4, 16, 64 or the whole tray, without running the
robot again. Each size becomes its own decision log (``<log>/blocks/n<size>/decisions.jsonl`` and frames) that
``benchmark.py replay`` and ``score`` read like any other, so every model gets the same pictures at every size.

A block's truth is over every capsule its picture shows: yes when a leaker lies wholly in it, no when none shows at
all, and none (left out of scoring) when a leaker shows only in part at an edge (tasks.capsules.block_truth). Size 1 asks leak_capsule on a crop centred on
the capsule, whose question says a neighbour's pool does not count.

    .\\run.ps1 scripts/capsule_block_curve.py --log output/capsules/<run> --sizes 1,4,16,64,0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from handoff.capsule_cell import TwoCells  # noqa: E402
from handoff.capsule_episode import CapsuleEpisode  # noqa: E402
from handoff.capsule_perception import SeenCapsule  # noqa: E402
from handoff.decider import QUESTIONS, Ladder  # noqa: E402
from handoff.logbook import read_decisions  # noqa: E402
from handoff.tasks.capsules import CapsuleContext, block_box, block_truth, blocks, capsule_box, crop, in_view, padded, state_text  # noqa: E402

PAD_M = 0.015  # the same pad the live scan uses


def scans(log: Path) -> list[dict]:
    """Every scan's tray record (the first leak_block of each scan carries it), with its episode."""
    out = []
    for d in read_decisions(log / "decisions.jsonl"):
        tray = d.get("context", {}).get("tray")
        if d["question"] == "leak_block" and tray and tray.get("tray_frame"):
            out.append({"episode": d["episode"], **tray})
    return out


def cut(log: Path, size: int, out: Path, max_scans: int | None = None) -> int:
    from PIL import Image

    cells = TwoCells()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for scan_no, scan in enumerate(scans(log)[:max_scans]):
        rgb = np.asarray(Image.open(log / scan["tray_frame"]).convert("RGB"))
        caps = [SeenCapsule(tuple(c["xyz"]), tuple(c["box"]), 0) for c in scan["capsules"]]
        leaky = [bool(c["leaker"]) for c in scan["capsules"]]
        pools = [c.get("pool_size") for c in scan["capsules"]]
        px_per_m = float(scan["px_per_m"])
        if size == 1:
            groups = [[c] for c in caps]
        elif size == 0:  # the whole tray in one picture
            groups = [caps]
        else:
            ctx = CapsuleContext(cells=cells, episode=CapsuleEpisode(0, scan["cell"], (), ()), ladder=Ladder([]), block=size, cell=scan["cell"])
            groups = blocks(caps, ctx)
        for group in groups:
            if size == 1:
                (cap,) = group
                i = caps.index(cap)
                picture = crop(rgb, capsule_box(cap, rgb, px_per_m), 0)
                qid, truth = "leak_capsule", "leaker" if leaky[i] else "good"
                shown = [i]
            else:
                window = padded(rgb, block_box(group), int(round(PAD_M * px_per_m)))
                shown = [caps.index(c) for c in in_view(caps, window)]
                picture = crop(rgb, block_box(group), int(round(PAD_M * px_per_m)))
                qid, truth = "leak_block", block_truth(caps, window, {i for i, x in enumerate(leaky) if x})
            name = f"frames/{scan['episode']}_s{scan_no:02d}_{len(rows):05d}_{qid}.png"
            Image.fromarray(np.ascontiguousarray(picture)).save(out / name)
            question = QUESTIONS[qid]
            rows.append({
                "episode": scan["episode"], "index": len(rows), "question": qid, "options": list(question.options),
                "state_text": state_text(None, qid), "frame": name, "rungs": [], "answer": None, "answered_by": None,
                "truth": truth, "context": {"block": size, "scan": scan_no, "capsules": [caps.index(c) for c in group], "in_view": shown,
                                            "pool_sizes": [pools[i] for i in shown if leaky[i]]},
            })
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True)
    parser.add_argument("--sizes", default="1,4,16,64,0", help="capsules per picture; 0 is the whole tray")
    parser.add_argument("--max-scans", type=int, default=None, help="only the first this many scans: every size then shows the same trays")
    args = parser.parse_args()
    log = Path(args.log)
    for size in (int(v) for v in args.sizes.split(",")):
        out = log / "blocks" / f"n{size if size else 'all'}"
        n = cut(log, size, out, args.max_scans)
        rows = [json.loads(line) for line in (out / "decisions.jsonl").open(encoding="utf-8")]
        yes = sum(1 for r in rows if r["truth"] in ("yes", "leaker"))
        unclear = sum(1 for r in rows if r["truth"] is None)
        print(json.dumps({"size": size or "whole tray", "pictures": n, "with_a_leaker": yes, "leaker_only_in_part": unclear, "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
