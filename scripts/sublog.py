"""Copy one question's decisions out of a decision log into a new log, with their frames and the arms already replayed
on them (Nov 6). For replaying a model on just those decisions without adding rows to the original log, which may be
selection data that a frozen protocol was chosen on.

    python3 scripts/sublog.py output/sort/test03 output/sort/test03_finish --question finish

Standard library only, so it runs from WSL's python3 or Windows' .venv alike.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; not overwriting it")
    (dst / "frames").mkdir(parents=True)
    keep, frames, renumber = [], 0, {}   # an arm row's index_global is the decision's line number in decisions.jsonl
    for i, line in enumerate((src / "decisions.jsonl").read_text(encoding="utf-8").splitlines()):
        d = json.loads(line)
        if d.get("question") != args.question:
            continue
        renumber[i] = len(keep)
        keep.append(line)
        if d.get("frame"):
            shutil.copy2(src / d["frame"], dst / d["frame"])
            frames += 1
    (dst / "decisions.jsonl").write_text("\n".join(keep) + "\n", encoding="utf-8")
    arms = 0
    if (src / "arms.jsonl").exists():
        with (dst / "arms.jsonl").open("w", encoding="utf-8") as out:
            for line in (src / "arms.jsonl").read_text(encoding="utf-8").splitlines():
                a = json.loads(line)
                if int(a["index_global"]) in renumber:
                    a["index_global"] = renumber[int(a["index_global"])]
                    out.write(json.dumps(a) + "\n")
                    arms += 1
    print(f"{len(keep)} {args.question} decisions, {frames} frames, {arms} arm rows -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
