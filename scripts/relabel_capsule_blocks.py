"""Recompute a capsule log's leak_block truths with the current rule, from what each logged picture shows (Nov 5).

A log made before e5ef177 labelled a block "yes" when any part of a leaker showed in it; the rule now is yes only when a
leaker lies wholly in the picture, None (left out of scoring) when it shows in part (tasks.capsules.block_truth). Every
leak_block row logs its pixel window and the capsules in view, and each scan's first block logs every capsule with its
truth, so the labels can be redone without running the robot. The original file is kept beside it once.

    python3 scripts/relabel_capsule_blocks.py output/capsules/sel01
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.capsule_perception import SeenCapsule  # noqa: E402
from handoff.tasks.capsules import block_truth  # noqa: E402


def relabel(log: Path) -> Counter:
    path = log / "decisions.jsonl"
    keep = log / "decisions.before_relabel.jsonl"
    if not keep.exists():
        shutil.copyfile(path, keep)
    # The current file, not the first copy: a log extended after an earlier relabel keeps its new rows (review round 7);
    # block_truth needs only each row's window and its scan's capsules, never the old label.
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    changes: Counter = Counter()
    tray = None
    for r in rows:
        ctx = r.get("context") or {}
        if r["question"] != "leak_block":
            continue
        if ctx.get("tray"):
            tray = ctx["tray"]
        if tray is None or "window" not in ctx:
            changes["no record"] += 1
            continue
        caps = [SeenCapsule(tuple(c["xyz"]), tuple(c["box"]), 0) for c in tray["capsules"]]
        leaky = {i for i, c in enumerate(tray["capsules"]) if c["leaker"]}
        new = block_truth(caps, tuple(ctx["window"]), leaky)
        changes[f"{r['truth']} -> {new}"] += 1
        r["truth"] = new
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return changes


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, dict(relabel(Path(arg))))
