r"""Can d3 tell a part landed in the wrong tray, once the prompt lets it say so?

The logged in_bin criteria define "no" as outside the bin, on the table, floor or in the hand, so a part in the other
tray has no "no" to pick (VALIDATION.md, "Prompt audit"). This re-asks d3 on saved frames under these prompts:

  baseline          the logged state and the production criteria (checks this run reproduces the logged behaviour)
  criteria          the logged state, with criteria whose "no" covers the other bin
  criteria_neutral  those criteria, and a state that names the target without saying the release was over it
  counts            the neutral state plus how many parts each tray held before this release (from the logged truth,
                    so the best case for what the robot's own record could supply)
  before_after      the neutral state and two frames: the trays after an earlier part's release, then now
  ba_swapped        control: the same text with the two frames in the opposite order
  ba_same_twice     control: the same text with the current frame sent twice, so nothing changed between them
  ba_one_frame      control: the same text with only the current frame

Nothing here changes the production question or the frozen ladder. Exploratory only: test03 chose the ladder, and
test04 is spent for claims, so it refuses test04.

    .\run.ps1 python scripts/probe_wrong_tray.py run
    .\run.ps1 python scripts/probe_wrong_tray.py score
    .\run.ps1 python scripts/probe_wrong_tray.py occupancy --logs test03 test04 scale-wrongbin    # no model: saved answers only

Resumable: each answer is appended the moment it returns and answered keys are skipped on a repeat; an errored answer
is retried. results.json, the completion marker, is written last. --force starts over.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handoff.decider import QUESTIONS, timed
from handoff.logbook import read_decisions

LOGS = ("test03", "scale-wrongbin")  # log02 and log02b left out: their state uses an older wording that never names the tray colours
REFUSED_LOGS = ("test04",)
OUT = Path("output/sort/wrong_tray_probe")
THRESHOLD = 0.80  # the frozen ladder's threshold (docs/PREREG.md)
GROUP_ORDER = ("wrong bin", "no bin", "intended bin")
FRAME_VARIANTS = ("before_after", "ba_swapped", "ba_same_twice", "ba_one_frame")
VARIANTS = ("baseline", "criteria", "criteria_neutral", "counts", *FRAME_VARIANTS)
BASELINE_TOLERANCE = 0.01
TRAYS = {"good": "green (right)", "rework": "orange (left)"}

BASE_QUESTION = QUESTIONS["in_bin"]
TARGET_CRITERIA = {
    "yes": "the part is resting inside the walls of the intended bin named above",
    "no": "the part is anywhere else: in the other bin, on the table, on the floor, or still in the hand",
}
COUNT_CRITERIA = {
    "yes": "the intended bin named above now holds one part more than it did before this release",
    "no": "it does not: the part went into the other bin, onto the table or floor, or is still in the hand",
}
RELEASE_CLAIM = re.compile(r"^The hand released the part over the (\w+) bin, the (.+?) tray, about ([\d.]+) s ago and is now (.+?)\. ")


def neutral_state(state_text: str) -> str:
    """Names the target without claiming the release was over it. Raises when the logged wording is not the expected one."""
    match = RELEASE_CLAIM.match(state_text)
    if match is None:
        raise ValueError(f"unexpected in_bin state: {state_text[:80]!r}")
    target, tray, seconds, hand = match.groups()
    rest = state_text[match.end():]
    return f"This part was meant for the {target} bin, the {tray} tray. The hand let go of it about {seconds} s ago and is now {hand}. {rest}"


def prompt_for(variant: str, state_text: str) -> tuple[object, str]:
    if variant == "baseline":
        return BASE_QUESTION, state_text
    question = replace(BASE_QUESTION, criteria=TARGET_CRITERIA)
    if variant == "criteria":
        return question, state_text
    if variant == "criteria_neutral":
        return question, neutral_state(state_text)
    raise ValueError(variant)


def count_state(state_text: str, counts: dict[str, int]) -> str:
    held = " and ".join(f"the {tray} tray held {counts.get(name, 0)} part{'' if counts.get(name, 0) == 1 else 's'}" for name, tray in TRAYS.items())
    return f"{neutral_state(state_text)} Before this release {held}."


def item_prompt(variant: str, item: dict) -> tuple[object, str]:
    if variant == "counts":
        return replace(BASE_QUESTION, criteria=COUNT_CRITERIA), count_state(item["state"], item["counts"])
    if variant in FRAME_VARIANTS:
        lead = "The first image shows the trays before this release and the second shows them now. "
        return replace(BASE_QUESTION, criteria=COUNT_CRITERIA), lead + neutral_state(item["state"])
    return prompt_for(variant, item["state"])


def applies(variant: str, item: dict) -> bool:
    return variant not in FRAME_VARIANTS or bool(item.get("before"))


def frames_for(variant: str, before: str, now: str) -> list[str]:
    """The images each frame variant sends, in order."""
    return {"before_after": [before, now], "ba_swapped": [now, before], "ba_same_twice": [now, now], "ba_one_frame": [now]}[variant]


def tray_counts(decision: dict) -> dict[str, int]:
    """Parts in each tray, not counting the part this decision is about."""
    ctx = decision.get("context", {})
    counts: Counter = Counter()
    for name, info in ctx.get("parts", {}).items():
        if name != ctx.get("part"):
            counts.update(info.get("in") or [])
    return dict(counts)


def where_part_was(decision: dict) -> str:
    ctx = decision.get("context", {})
    bins_in = ctx.get("parts", {}).get(ctx.get("part"), {}).get("in") or []
    if not bins_in:
        return "no bin"
    return "intended bin" if ctx.get("intended_bin") in bins_in else "wrong bin"


def collect(root: Path, logs: tuple[str, ...]) -> tuple[list[dict], Counter]:
    """Every in_bin decision that has a frame on disk, wrong-bin first. Second value counts what was left out and why."""
    items, skipped = [], Counter()
    for log in logs:
        if log in REFUSED_LOGS:
            raise SystemExit(f"{log} is spent for claims; a new claim needs test05")
        last_frame: dict[tuple, str] = {}  # (episode, part) -> the part's latest in_bin frame so far
        for index, decision in enumerate(read_decisions(root / log / "decisions.jsonl")):
            if decision.get("question") != "in_bin":
                continue
            episode, part = decision.get("episode"), decision["context"].get("part")
            earlier = [path for (ep, other), path in last_frame.items() if ep == episode and other != part]
            frame = decision.get("frame")
            if not frame or not (root / log / frame).exists():
                skipped[f"{log}: no frame"] += 1
                continue
            items.append({
                "key": f"{log}/{index}", "log": log, "frame": str(root / log / frame), "state": decision["state_text"],
                "group": where_part_was(decision), "truth": decision.get("truth"),
                "landing": f"{log}/{decision.get('episode')}/{decision['context'].get('part')}",
                "counts": tray_counts(decision), "target_occupied": tray_counts(decision).get(decision["context"].get("intended_bin"), 0) > 0,
                "before": earlier[-1] if earlier else None,
            })
            last_frame[(episode, part)] = str(root / log / frame)
    items.sort(key=lambda item: GROUP_ORDER.index(item["group"]))
    return items, skipped


def read_answers(path: Path) -> dict[tuple[str, str], dict]:
    """Latest usable answer per (key, variant); an errored line never counts as answered."""
    answers: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    answers[(row["key"], row["variant"])] = row
    return answers


def outcome(probs: dict[str, float], threshold: float = THRESHOLD) -> str:
    """What the ladder would do: act on an answer at or above the threshold, else hand off."""
    top = max(probs, key=probs.__getitem__)
    return f"said {top}" if probs[top] >= threshold else "handed off"


def score(items: list[dict], answers: dict[tuple[str, str], dict], threshold: float = THRESHOLD) -> dict:
    """Per variant and group: decisions, distinct landings, outcomes, and how many are still unanswered."""
    table: dict = {}
    for variant in VARIANTS:
        table[variant] = {}
        for group in GROUP_ORDER:
            members = [item for item in items if item["group"] == group and applies(variant, item)]
            counts: Counter = Counter()
            landings = set()
            missing = 0
            for item in members:
                row = answers.get((item["key"], variant))
                if row is None:
                    missing += 1
                    continue
                counts[outcome(row["probs"], threshold)] += 1
                landings.add(item["landing"])
            table[variant][group] = {
                "decisions": len(members), "answered_landings": len(landings), "missing": missing,
                "said yes": counts["said yes"], "handed off": counts["handed off"], "said no": counts["said no"],
            }
    return table


def ask_two_frames(url: str, question, state: str, images: list[str], post, name: str = "d3-27b-vision"):
    def call() -> tuple[dict[str, float], dict]:
        body = {"state": state, "images": [f"data:image/png;base64,{image}" for image in images],
                "questions": {question.id: {"type": "choice", "instructions": question.text, "criteria": dict(question.criteria)}}}
        return post(f"{url}/v1/systemone", body, 120.0)["answers"][question.id]["probabilities"], {}

    return timed(name, question.options, call)


def occupancy(decisions: list[dict], logged_yes: dict[int, float], threshold: float = THRESHOLD) -> dict[tuple[str, str], dict[str, int]]:
    """Wrong-bin decisions and distinct landings, split by whether the target tray already held another part and what d3 did."""
    cells: dict[tuple[str, str], list] = {}
    for index, decision in enumerate(decisions):
        if decision.get("question") == "in_bin" and where_part_was(decision) == "wrong bin" and index in logged_yes:
            occupied = tray_counts(decision).get(decision["context"].get("intended_bin"), 0) > 0
            cell = ("target occupied" if occupied else "target empty", outcome({"yes": logged_yes[index], "no": 1 - logged_yes[index]}, threshold))
            cells.setdefault(cell, []).append((decision.get("episode"), decision["context"].get("part")))
    return {cell: {"decisions": len(landings), "landings": len(set(landings))} for cell, landings in cells.items()}


def logged_arm_yes(arms: Path) -> dict[int, float]:
    """d3's in_bin p(yes) by decision index, from the log's own replay."""
    logged: dict[int, float] = {}
    if arms.exists():
        for line in arms.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("arm") == "d3-27b-vision" and row.get("question") == "in_bin" and not row.get("error"):
                logged[row["index_global"]] = row["probs"]["yes"]
    return logged


def baseline_drift(answers: dict[tuple[str, str], dict], log: str, logged: dict[int, float]) -> tuple[int, float]:
    """How many baseline answers can be compared with the logged run, and the largest difference."""
    diffs = [abs(row["probs"]["yes"] - logged[int(key.split("/")[1])]) for (key, variant), row in answers.items()
             if variant == "baseline" and key.startswith(f"{log}/") and int(key.split("/")[1]) in logged]
    return len(diffs), max(diffs, default=0.0)


def print_table(table: dict) -> None:
    for variant, groups in table.items():
        print(variant)
        for group, row in groups.items():
            print(f"  {group:13s} n={row['decisions']:4d} landings={row['answered_landings']:4d} yes={row['said yes']:4d} "
                  f"handed off={row['handed off']:4d} no={row['said no']:4d} missing={row['missing']}")


def run(args: argparse.Namespace) -> int:
    from handoff.rungs import DeciderRung, _post
    from handoff.servers import D3_URL, ensure_d3

    root = Path(args.root)
    out = root / OUT.name
    out.mkdir(parents=True, exist_ok=True)
    answers_path, marker = out / "answers.jsonl", out / "results.json"
    if args.force:
        answers_path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    items, skipped = collect(root, tuple(args.logs))
    print(f"{len(items)} in_bin decisions, {dict(Counter(item['group'] for item in items))}; left out: {dict(skipped)}", flush=True)
    print(ensure_d3(), flush=True)
    d3 = DeciderRung("d3-27b-vision", D3_URL, with_image=True, data_url=True, timeout=120.0, expect=("d3-27b",))
    print("served:", d3.check_model(), flush=True)
    answers = read_answers(answers_path)
    with answers_path.open("a", encoding="utf-8") as handle:
        for item in items:
            image = base64.b64encode(Path(item["frame"]).read_bytes()).decode()
            for variant in VARIANTS:
                if (item["key"], variant) in answers or not applies(variant, item):
                    continue
                question, state = item_prompt(variant, item)
                if variant in FRAME_VARIANTS:
                    before = base64.b64encode(Path(item["before"]).read_bytes()).decode()
                    answer = ask_two_frames(D3_URL, question, state, frames_for(variant, before, image), _post)
                else:
                    answer = d3.answer(question, state, image)
                row = {"key": item["key"], "variant": variant, "group": item["group"], "probs": answer.probs,
                       "latency_ms": answer.latency_ms, "error": answer.error}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if not answer.error:
                    answers[(item["key"], variant)] = row
                print(f"{item['group']:13s} {item['key']:22s} {variant:17s} "
                      f"{answer.error or answer.top + f' {answer.confidence:.2f}'}", flush=True)
    return finish(items, answers, marker, root)


def finish(items: list[dict], answers: dict, marker: Path, root: Path | None = None) -> int:
    if root is not None:
        for log in sorted({item["log"] for item in items}):
            compared, largest = baseline_drift(answers, log, logged_arm_yes(root / log / "arms.jsonl"))
            print(f"baseline against the logged run, {log}: {compared} compared, largest difference {largest:.4f}")
            if largest > BASELINE_TOLERANCE:
                print("baseline does not reproduce the logged run: results.json not written")
                return 1
    table = score(items, answers)
    print_table(table)
    wrong = [item for item in items if item["group"] == "wrong bin"]
    for label, chosen in (("target tray already held a part", True), ("target tray empty", False)):
        print(f"wrong bin only, {label}:")
        for variant, groups in score([item for item in wrong if item.get("target_occupied") == chosen], answers).items():
            row = groups["wrong bin"]
            print(f"  {variant:17s} n={row['decisions']:3d} yes={row['said yes']:3d} handed off={row['handed off']:3d} no={row['said no']:3d} missing={row['missing']}")
    missing = sum(row["missing"] for groups in table.values() for row in groups.values())
    if missing:
        print(f"{missing} answers missing: run again; results.json not written")
        return 1
    marker.write_text(json.dumps({"threshold": THRESHOLD, "table": table}, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "score", "occupancy"))
    parser.add_argument("--root", default="output/sort")
    parser.add_argument("--logs", nargs="+", default=list(LOGS))
    parser.add_argument("--force", action="store_true", help="discard saved answers and start over")
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    root = Path(args.root)
    if args.command == "occupancy":
        saved = read_answers(root / OUT.name / "answers.jsonl")
        for log in args.logs:
            decisions = read_decisions(root / log / "decisions.jsonl")
            logged, source = logged_arm_yes(root / log / "arms.jsonl"), "logged replay"
            if not logged:
                logged = {int(key.split("/")[1]): row["probs"]["yes"] for (key, variant), row in saved.items() if variant == "baseline" and key.startswith(f"{log}/")}
                source = "this probe's baseline re-ask"
            if not logged:
                print(f"{log}: no d3 answers saved")
                return 1
            print(log, f"({source})", dict(sorted(occupancy(decisions, logged).items())))
        return 0
    items, _ = collect(root, tuple(args.logs))
    return finish(items, read_answers(root / OUT.name / "answers.jsonl"), root / OUT.name / "results.json", root)


if __name__ == "__main__":
    raise SystemExit(main())
