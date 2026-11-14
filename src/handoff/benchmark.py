"""Benchmark arithmetic over recorded decisions.

Everything here works on plain dicts read from the logs, so it runs without a GPU:

- a **recorded decision** (from ``decisions.jsonl``): question, options, state text, frame, truth;
- an **arm answer** (from ``arms.jsonl``): for one decision and one arm, the probabilities per
  option and the latency. The replay script fills these in, one arm at a time, so the models
  never have to share the GPU.

A **ladder** is composed offline from arm answers: climb the rungs in order, accept the first
whose confidence meets its threshold, otherwise fall to the person (who is right, at a cost).
Thresholds are chosen on a held-out split and reported on the rest.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CONFIDENT = 0.9  # the Jev convention's "act without a check" band

# Questions a hard rule from sim state decides before any model is asked (safe_to_grasp: a part on the floor or a
# neighbour within 6 cm forces abort). Every arm sits at the base rate on them, so the report also scores the
# "judgement" decisions on their own: the ones where a model's answer is what the robot acts on.
RULE_DECIDED = frozenset({"safe_to_grasp"})
BOOTSTRAP = 1000  # resamples for the by-episode intervals


def relabel_bin_questions(decisions: list[dict[str, Any]], layout: Any, part_size: tuple[float, float, float]) -> int:
    """Recompute in_bin and sort_complete truths from the logged part positions with the current check.

    The log records every part's position at those questions, so a fix to the geometry check can
    be applied to old logs without re-running the robot. Returns how many labels changed.
    """
    import numpy as np

    from .checks import WorldState, in_bin
    from .sim.base import ObjectState, Pose

    from dataclasses import replace

    changed = 0
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    for d in decisions:
        parts = d.get("context", {}).get("parts")
        if d["question"] not in ("in_bin", "sort_complete") or not parts:
            continue
        # Logs made since the bins move per episode record where they truly were; older logs had them where the layout says.
        true_xy = d["context"].get("bins_true")
        bins = (replace(layout, bin_xy=tuple((n, x, y) for n, (x, y) in true_xy.items())) if true_xy else layout).bin_regions()
        state = WorldState(
            t=0.0,
            objects={p: ObjectState(np.array(v["pos"], dtype=float), identity, np.zeros(3)) for p, v in parts.items()},
            wrists={}, hand_closure={}, contacts={}, table_top=layout.table_top, bins=bins, part_size=part_size,
        )
        if d["question"] == "in_bin":
            intended = d["context"].get("intended_bin")
            truth = "yes" if intended in bins and in_bin(state, d["context"]["part"], intended) else "no"
        else:
            truth = "yes" if all(any(in_bin(state, p, b) for b in bins) for p in parts) else "no"
        if truth != d.get("truth"):
            d["truth"] = truth
            changed += 1
    return changed


def top(probs: Mapping[str, float]) -> tuple[str, float]:
    best = max(probs, key=probs.__getitem__)
    return best, float(probs[best])


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], bins: int = 10) -> float:
    """Average gap between stated confidence and measured accuracy, weighted by bin size."""
    conf = np.asarray(list(confidences), dtype=float)
    hit = np.asarray(list(correct), dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return float(ece)


@dataclass
class ArmReport:
    arm: str
    n: int
    accuracy: float
    confidently_wrong: float  # share of all decisions answered with confidence >= 0.9 and wrong
    ece: float
    mean_confidence: float
    latency_ms: float
    human_rate: float = 0.0
    alone_right: float = 0.0  # share of all decisions answered without the person AND right (= accuracy for a plain arm)
    by_question: dict[str, dict[str, float]] | None = None
    judgement: dict[str, float] | None = None  # the same rates on the decisions no hard rule decides (see RULE_DECIDED)
    interval: dict[str, tuple[float, float]] | None = None  # 95% by-episode bootstrap: accuracy, confidently_wrong, human_rate, alone_right, judgement_*

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The headline rates of a set of scoring rows (accuracy counts the person's answers; alone_right does not)."""
    n = len(rows)
    if n == 0:
        return {"accuracy": float("nan"), "confidently_wrong": float("nan"), "human_rate": float("nan"), "alone_right": float("nan")}
    correct = [r["answer"] == r["truth"] for r in rows]
    cw = sum(1 for r, c in zip(rows, correct, strict=True) if not c and float(r["confidence"]) >= CONFIDENT and not r.get("human"))
    human = sum(1 for r in rows if r.get("human"))
    alone_right = sum(1 for r, c in zip(rows, correct, strict=True) if c and not r.get("human"))
    return {"accuracy": float(np.mean(correct)), "confidently_wrong": cw / n, "human_rate": human / n, "alone_right": alone_right / n}


def bootstrap_by_episode(rows: list[dict[str, Any]], n_resamples: int = BOOTSTRAP, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% intervals for :func:`_rates` from resampling EPISODES with replacement, not decisions.

    Decisions inside one episode share the parts, the bins and the camera, so they are not independent draws;
    the episode is the unit. Rows without an ``episode`` key are treated as one episode each. Returns an empty
    dict when there is only one episode: no interval can be made from it.
    """
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.get("episode", f"row{i}")].append(r)
    episodes = list(groups.values())
    if len(episodes) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        picked = rng.integers(0, len(episodes), size=len(episodes))
        resampled = [r for k in picked for r in episodes[k]]
        for key, v in _rates(resampled).items():
            samples[key].append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}


def score(arm: str, rows: list[dict[str, Any]], *, bootstrap: int = BOOTSTRAP, seed: int = 0) -> ArmReport:
    """``rows``: one per decision with keys answer, confidence, truth, latency_ms, human (bool), question, episode.

    ``bootstrap`` resamples give the 95% by-episode intervals (0 skips them).
    """
    n = len(rows)
    if n == 0:
        return ArmReport(arm, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    correct = [r["answer"] == r["truth"] for r in rows]
    conf = [float(r["confidence"]) for r in rows]
    rates = _rates(rows)
    judgement_rows = [r for r in rows if r["question"] not in RULE_DECIDED]
    judgement = {"n": len(judgement_rows), **_rates(judgement_rows)} if judgement_rows else None
    interval: dict[str, tuple[float, float]] | None = None
    if bootstrap:
        interval = bootstrap_by_episode(rows, bootstrap, seed)
        if judgement_rows:
            interval.update({f"judgement_{k}": v for k, v in bootstrap_by_episode(judgement_rows, bootstrap, seed).items()})
    by_q: dict[str, dict[str, float]] = {}
    for q in sorted({r["question"] for r in rows}):
        sub = [(r, c) for r, c in zip(rows, correct, strict=True) if r["question"] == q]
        by_q[q] = {
            "n": len(sub),
            "accuracy": round(float(np.mean([c for _, c in sub])), 4),
            "confidently_wrong": round(sum(1 for r, c in sub if not c and float(r["confidence"]) >= CONFIDENT and not r.get("human")) / len(sub), 4),
            "human_rate": round(sum(1 for r, _ in sub if r.get("human")) / len(sub), 4),
        }
    # ECE over the machine's answers only: the person's rows are certain and right by construction and would pull a ladder's ECE toward 0.
    machine = [i for i, r in enumerate(rows) if not r.get("human")]
    return ArmReport(
        arm=arm,
        n=n,
        accuracy=rates["accuracy"],
        confidently_wrong=rates["confidently_wrong"],
        ece=expected_calibration_error([conf[i] for i in machine], [correct[i] for i in machine]),
        mean_confidence=float(np.mean(conf)),
        latency_ms=float(np.mean([float(r.get("latency_ms", 0.0)) for r in rows])),
        human_rate=rates["human_rate"],
        alone_right=rates["alone_right"],
        by_question=by_q,
        judgement=judgement,
        interval=interval or None,
    )


def _ms(answer: Mapping[str, Any]) -> float:
    """A recorded latency; NaN when the arm has none (a run on Kaggle's servers reports no per-call time)."""
    value = answer.get("latency_ms", 0.0)
    return float("nan") if value is None else float(value)


def rows_for_arm(decisions: list[dict[str, Any]], answers: Mapping[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn one arm's recorded answers into scoring rows (decisions it has no answer for are skipped)."""
    out = []
    for d in decisions:
        a = answers.get(int(d["index_global"]))
        if a is None or d.get("truth") is None:
            continue
        choice, conf = top(a["probs"])
        out.append({"question": d["question"], "episode": d.get("episode"), "answer": choice, "confidence": conf, "truth": d["truth"], "latency_ms": _ms(a), "human": False})
    return out


Threshold = float | Mapping[str, float]  # one number, or one per question id


def threshold_for(threshold: Threshold, question: str) -> float:
    if isinstance(threshold, Mapping):
        return float(threshold.get(question, 1.0))
    return float(threshold)


def compose_ladder(
    decisions: list[dict[str, Any]],
    rung_answers: list[tuple[str, Mapping[int, dict[str, Any]], Threshold]],
    *,
    human_latency_ms: float = 5000.0,
) -> list[dict[str, Any]]:
    """Climb recorded rungs (name, answers by decision index, threshold); the person answers the rest.

    A threshold is one number for the rung, or a mapping from question id to number, since a rung
    can be trustworthy on one question and not on another.
    """
    out = []
    for d in decisions:
        if d.get("truth") is None:
            continue
        idx = int(d["index_global"])
        latency = 0.0
        row = None
        for name, answers, threshold in rung_answers:
            a = answers.get(idx)
            if a is None:
                continue
            latency += _ms(a)
            choice, conf = top(a["probs"])
            if conf >= threshold_for(threshold, d["question"]):
                row = {"question": d["question"], "episode": d.get("episode"), "answer": choice, "confidence": conf, "truth": d["truth"], "latency_ms": latency, "human": False, "rung": name}
                break
        if row is None:
            row = {"question": d["question"], "episode": d.get("episode"), "answer": d["truth"], "confidence": 1.0, "truth": d["truth"], "latency_ms": latency + human_latency_ms, "human": True, "rung": "human"}
        out.append(row)
    return out


def choose_thresholds(
    decisions: list[dict[str, Any]],
    rungs: list[tuple[str, Mapping[int, dict[str, Any]]]],
    *,
    grid: Iterable[float] = tuple(np.round(np.arange(0.3, 1.0001, 0.05), 2)),
    human_cost: float = 0.5,
) -> dict[str, float]:
    """Pick each rung's threshold greedily, lowest rung first, on the given (held-out) decisions.

    The objective per rung: a right answer scores +1, a wrong answer -1, and a question passed up
    scores ``1 - human_cost`` (the person answers it right, at a cost). So a rung is trusted on a
    band of confidence only where its accuracy there beats ``1 - human_cost / 2`` (0.75 at the
    default). A rung whose best threshold is 1.0 never answers: it earns no place on the ladder.
    """
    chosen: dict[str, float] = {}
    remaining = [d for d in decisions if d.get("truth") is not None]
    for name, answers in rungs:
        if not remaining:  # the rungs below answered everything: this one never gets asked
            chosen[name] = 1.0
            continue
        best_t, best_score = 1.0, -1.0
        for t in grid:
            answered = [d for d in remaining if int(d["index_global"]) in answers and top(answers[int(d["index_global"])]["probs"])[1] >= t]
            right = sum(1 for d in answered if top(answers[int(d["index_global"])]["probs"])[0] == d["truth"])
            wrong = len(answered) - right
            passed_up = len(remaining) - len(answered)
            s = (right - wrong + (1.0 - human_cost) * passed_up) / len(remaining)
            if s > best_score + 1e-12:
                best_t, best_score = float(t), s
        chosen[name] = best_t
        remaining = [d for d in remaining if not (int(d["index_global"]) in answers and top(answers[int(d["index_global"])]["probs"])[1] >= best_t)]
    return chosen


def choose_thresholds_per_question(
    decisions: list[dict[str, Any]],
    rungs: list[tuple[str, Mapping[int, dict[str, Any]]]],
    *,
    min_decisions: int = 0,
    **kwargs: Any,
) -> dict[str, dict[str, float]]:
    """:func:`choose_thresholds` run separately for each question id: rung -> question -> threshold.

    A question with fewer than ``min_decisions`` labelled decisions to fit on takes the rung's single threshold
    (fitted on all the decisions) instead: a cut-off chosen on a few dozen answers is mostly noise.
    """
    out: dict[str, dict[str, float]] = {name: {} for name, _ in rungs}
    pooled = choose_thresholds(decisions, rungs, **kwargs) if min_decisions > 0 else {}
    for q in sorted({d["question"] for d in decisions}):
        mine = [d for d in decisions if d["question"] == q]
        if sum(1 for d in mine if d.get("truth") is not None) < min_decisions:
            for name, t in pooled.items():
                out[name][q] = t
            continue
        chosen = choose_thresholds(mine, rungs, **kwargs)
        for name, t in chosen.items():
            out[name][q] = t
    return out


def ladder_objective(decisions: list[dict[str, Any]], rungs: list[tuple[str, Mapping[int, dict[str, Any]]]], thresholds: Mapping[str, float], *, human_cost: float = 0.5) -> float:
    """The objective :func:`choose_thresholds` maximises, scored for the whole ladder at once: right +1, wrong -1, person ``1 - human_cost``."""
    total, n = 0.0, 0
    for d in decisions:
        if d.get("truth") is None:
            continue
        n += 1
        idx = int(d["index_global"])
        for name, answers in rungs:
            a = answers.get(idx)
            if a is None:
                continue
            choice, conf = top(a["probs"])
            if conf >= thresholds[name]:
                total += 1.0 if choice == d["truth"] else -1.0
                break
        else:
            total += 1.0 - human_cost
    return total / n if n else 0.0


def choose_thresholds_joint(
    decisions: list[dict[str, Any]],
    rungs: list[tuple[str, Mapping[int, dict[str, Any]]]],
    *,
    grid: Iterable[float] = tuple(np.round(np.arange(0.3, 1.0001, 0.05), 2)),
    human_cost: float = 0.5,
) -> dict[str, float]:
    """Every rung's threshold searched together, so a lower rung's cut-off knows what the rungs above it really do.

    :func:`choose_thresholds` fits each rung as if what it passes up goes straight to the person. The per-choice
    rule of arXiv 2610.21412 (Section 3) accepts a cheap answer only when its confidence beats the accuracy of
    whatever answers instead, which here is the next rung, not the person. An exhaustive search over the grid on
    the same objective does that exactly; the greedy answer is one point of the search, so it never scores lower
    on the decisions it is fitted on. Ties keep the higher thresholds (fewer unchecked answers).
    """
    grid = tuple(float(t) for t in grid)
    names = [name for name, _ in rungs]
    best: dict[str, float] = dict.fromkeys(names, 1.0)
    best_score = ladder_objective(decisions, rungs, best, human_cost=human_cost)
    for combo in itertools.product(sorted(grid, reverse=True), repeat=len(names)):
        cand = dict(zip(names, combo, strict=True))
        s = ladder_objective(decisions, rungs, cand, human_cost=human_cost)
        if s > best_score + 1e-12:
            best, best_score = cand, s
    return best


def choose_thresholds_joint_per_question(
    decisions: list[dict[str, Any]],
    rungs: list[tuple[str, Mapping[int, dict[str, Any]]]],
    **kwargs: Any,
) -> dict[str, dict[str, float]]:
    """:func:`choose_thresholds_joint` run separately for each question id: rung -> question -> threshold."""
    out: dict[str, dict[str, float]] = {name: {} for name, _ in rungs}
    for q in sorted({d["question"] for d in decisions}):
        chosen = choose_thresholds_joint([d for d in decisions if d["question"] == q], rungs, **kwargs)
        for name, t in chosen.items():
            out[name][q] = t
    return out


def rung_value(decisions: list[dict[str, Any]], lower: Mapping[int, dict[str, Any]], lower_t: Threshold, upper: Mapping[int, dict[str, Any]]) -> dict[str, float]:
    """On the decisions the lower rung escalates, how often is the upper rung right vs the lower's own guess?"""
    escalated = [d for d in decisions if d.get("truth") is not None and int(d["index_global"]) in lower and top(lower[int(d["index_global"])]["probs"])[1] < threshold_for(lower_t, d["question"])]
    with_upper = [d for d in escalated if int(d["index_global"]) in upper]
    if not with_upper:
        return {"n_escalated": len(escalated), "upper_accuracy": float("nan"), "lower_accuracy_on_same": float("nan")}
    up = float(np.mean([top(upper[int(d["index_global"])]["probs"])[0] == d["truth"] for d in with_upper]))
    low = float(np.mean([top(lower[int(d["index_global"])]["probs"])[0] == d["truth"] for d in with_upper]))
    return {"n_escalated": len(with_upper), "upper_accuracy": up, "lower_accuracy_on_same": low}


def split(decisions: list[dict[str, Any]], held_out_fraction: float = 0.3, seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Held-out (for choosing thresholds) and test (for reporting), split by episode so frames never leak."""
    episodes = sorted({d["episode"] for d in decisions})
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_held = max(1, int(round(len(episodes) * held_out_fraction)))
    held = set(episodes[:n_held])
    return [d for d in decisions if d["episode"] in held], [d for d in decisions if d["episode"] not in held]


def collapse_finish(decisions: list[dict[str, Any]], arms: dict[str, dict[int, dict[str, Any]]]) -> int:
    """Fold logs that asked good / scratched / unpainted into today's good / defective. Returns how many truths changed.

    The finish question had three answers until Oct 12. A model's probability for "defective" is taken as the sum of
    what it gave the two defect labels, which is what it would have routed to the rework bin.
    """
    old = ("scratched", "unpainted")
    changed = 0
    for d in decisions:
        if d.get("question") != "finish":
            continue
        if d.get("truth") in old:
            d["truth"] = "defective"
            changed += 1
        if d.get("answer") in old:
            d["answer"] = "defective"
        if d.get("options") and any(o in old for o in d["options"]):
            d["options"] = ["good", "defective"]
    for rows in arms.values():
        for r in rows.values():
            probs = r.get("probs") or {}
            if any(o in probs for o in old):
                r["probs"] = {"good": float(probs.get("good", 0.0)), "defective": float(sum(probs.get(o, 0.0) for o in old))}
    return changed


def group_by_arm(arm_rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in arm_rows:
        out[r["arm"]][int(r["index_global"])] = r
    return dict(out)


SCALE_WINDOW_MS = 200.0  # a tray reading averages 0.2 s (handoff/sensors.py)


def scale_answers(decisions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """The tray-scale arm, from the readings logged with each in_bin decision (``context.scale``), no replay needed.

    A settled verdict is certain (probability 1); an unclear one is 0.5 each, so at threshold 1.0 it climbs on to the
    next rung. Decisions logged without scales have no answer here and go straight to the next rung.
    """
    out: dict[int, dict[str, Any]] = {}
    for d in decisions:
        scale = (d.get("context") or {}).get("scale")
        if d["question"] != "in_bin" or not scale:
            continue
        verdict = scale.get("verdict")
        probs = {o: (1.0 if o == verdict else 0.0) for o in ("yes", "no")} if verdict in ("yes", "no") else {"yes": 0.5, "no": 0.5}
        out[int(d["index_global"])] = {"arm": "scale", "index_global": int(d["index_global"]), "probs": probs, "latency_ms": SCALE_WINDOW_MS, "where": scale.get("where")}
    return out


def add_logged_arms(decisions: list[dict[str, Any]], arms: dict[str, dict[int, dict[str, Any]]]) -> list[str]:
    """Add the arms a log carries in its decisions rather than in arms.jsonl (today the tray scales, from a ``--scales``
    run), so ``benchmark.py score`` reports them beside the replayed arms. Returns the names added.

    The scale arm answers in_bin only, so the 95% coverage check keeps it out of every ladder: it is scored as an arm,
    not placed on the frozen ladder (docs/PREREG.md), which it is not part of."""
    scale = scale_answers(decisions)
    if not scale:
        return []
    arms["scale"] = scale
    return ["scale"]
