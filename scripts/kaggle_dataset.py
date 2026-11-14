"""Build the folder for a private Kaggle dataset of logged decisions, for arms that run on Kaggle Benchmarks.

Each row carries the exact prompt the local self-confidence arms get, the frame file and the (relabelled) truth,
so a Kaggle-run arm answers the same question the same way and its rows score like any other arm.

    bash run.sh scripts/kaggle_dataset.py --out output/kaggle/handoff-frames \
        --log output/sort/test03:18 --log output/sort/log02b:all --id duplonicus/handoff-frames

`--log path:N` takes N whole episodes picked with a fixed seed (whole episodes keep the by-episode bootstrap
honest); `path:all` takes every decision. Upload with `scripts/kaggle.sh datasets create -p <out>` (private by
default), or `datasets version -p <out> -m ...` for a new version.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import load  # noqa: E402
from handoff.decider import QUESTIONS  # noqa: E402
from handoff.rungs import self_confidence_prompt  # noqa: E402


def pick(decisions: list[dict], how: str, seed: int) -> list[dict]:
    if how == "all":
        return decisions
    episodes = sorted({d["episode"] for d in decisions})
    chosen = set(random.Random(seed).sample(episodes, int(how)))
    return [d for d in decisions if d["episode"] in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", action="append", required=True, help="path:N (N whole episodes) or path:all")
    parser.add_argument("--id", required=True, help="owner/dataset-slug")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    n = 0
    with open(out / "decisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["log", "index_global", "episode", "question", "prompt", "frame", "options", "truth"])
        for spec in args.log:
            path, _, how = spec.rpartition(":")
            log = Path(path)
            rows = [d for d in pick(load(log), how, args.seed) if d.get("frame") and d.get("truth")]
            for d in rows:
                q = QUESTIONS[d["question"]]
                src = log / d["frame"]
                shutil.copy(src, out / src.name)
                w.writerow([log.name, d["index_global"], d["episode"], d["question"], self_confidence_prompt(q, d["state_text"]), src.name, json.dumps(list(q.options)), d["truth"]])
            counts = {qid: sum(1 for d in rows if d["question"] == qid) for qid in QUESTIONS}
            print(f"{log.name}: {len(rows)} decisions from {len({d['episode'] for d in rows})} episodes {counts}")
            n += len(rows)
    title = args.id.split("/")[1].replace("-", " ")
    (out / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": args.id, "licenses": [{"name": "other"}]}), encoding="utf-8")
    print(f"{n} decisions in {out}")


if __name__ == "__main__":
    main()
