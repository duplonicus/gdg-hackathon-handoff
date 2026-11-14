"""Questions, rungs and the ladder.

A **question** has a fixed, typed list of options; every rung sees the same list. A **rung** is
anything that can turn (question, state text, camera frame) into a probability per option: a
local decision model, a bigger local one, a cloud model, or a person. The **ladder** asks the
rungs in order and accepts the first answer that is confident enough *and* executable in the
current world state; if none is, the last rung (a person, or the question's safe option) decides.

Confidence of a choice is the top option's probability, as the Decider's docs and the public Jev
convention define it. Thresholds are per rung and must be measured, not assumed.

A **rule** is a hard check from sim state that overrides every rung for safety-type questions
("should anything happen at all?"), the kind the closest prior work found decision models most
confidently wrong on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[str, ...]
    criteria: Mapping[str, str]  # one line per option, shown to models and people alike
    kind: str = "judgement"  # "judgement" (what is this?) or "safety" (should anything happen?)
    safe_option: str | None = None  # chosen when nobody is confident and no person answers

    def __post_init__(self) -> None:
        assert set(self.criteria) == set(self.options), self.id
        assert self.safe_option is None or self.safe_option in self.options


QUESTIONS: dict[str, Question] = {
    "finish": Question(
        "finish",
        "Is the paint finish of the part in view good or defective?",
        # Two answers, not three (dup, Oct 12): which kind of defect it is does not change what the robot does, and a
        # person looking at the frames could not tell "scratched" from "unpainted" either.
        ("good", "defective"),
        {"good": "every visible face is one uniform colour of paint, with no marks", "defective": "anything else: lines, patches or areas of another colour on any face"},
        # A part nobody vouched for goes to the rework bin, never to the good bin. Without this the fallback was the
        # first option, "good", and an unanswered operator question shipped a defective part (seen Oct 11).
        safe_option="defective",
    ),
    "safe_to_grasp": Question(
        "safe_to_grasp",
        "Should the robot close its hand on the part now?",
        ("grasp", "re_look", "ask_operator", "abort"),
        {"grasp": "the part is in the hand's reach and nothing is in the way", "re_look": "the view is unclear, look again", "ask_operator": "a person should decide", "abort": "stop, something is wrong"},
        kind="safety",
        safe_option="ask_operator",
    ),
    "in_bin": Question(
        "in_bin",
        "Did the part land inside the intended bin?",
        ("yes", "no"),
        {"yes": "the part is resting inside the bin walls", "no": "the part is outside the bin, on the table, floor, or still in the hand"},
        kind="judgement",
        safe_option="no",
    ),
    "sort_complete": Question(
        "sort_complete",
        "Is the sort finished: every part in a bin and the table clear?",
        ("yes", "no"),
        {"yes": "no parts remain on the table and every bin holds its parts", "no": "a part is still on the table or somewhere else"},
        kind="safety",
        safe_option="no",
    ),
    # ----- the soft-gel inspection cells (Nov 4; dup's capsule plant). A leaker has a pool of its fill on the parchment
    # around it; a good capsule has none, and the parchment is matte, so nothing else on it looks like a pool. -----
    "where_needed": Question(
        "where_needed",
        "Which cell needs the robot now: the one in the left picture, the one in the right picture, or neither?",
        # Options with distinct first words: "cell_a" and "cell_b" both begin with the token "cell", so a model read by
        # its first token's probability could not tell them apart (Qwen answered cell_b with 0.92 on cell_a, Nov 4).
        ("left", "right", "neither"),
        {
            "left": "the cell in the left picture has work: a tray of capsules on its table or full trays in its cart",
            "right": "the cell in the right picture has work: a tray of capsules on its table or full trays in its cart",
            "neither": "both cells look finished: empty trays in the carts and no capsules on the tables",
        },
        safe_option="neither",  # unsure where to go: stay put and let a person send it
    ),
    "leak_block": Question(
        "leak_block",
        "Is any capsule in this picture a leaker?",
        ("yes", "no"),
        {
            "yes": "at least one capsule sits in a pool of its fill: a glossy amber patch on the paper around it, wider than the capsule",
            "no": "every capsule sits on clean, dry paper",
        },
        safe_option="yes",  # unsure: look at each capsule in it
    ),
    "leak_capsule": Question(
        "leak_capsule",
        "Is the capsule in the middle of this picture a leaker?",
        ("leaker", "good"),
        {
            "leaker": "the capsule in the middle sits in a pool of its fill on the paper",
            "good": "the paper around the capsule in the middle is clean and dry; a pool around a neighbour does not count",
        },
        safe_option="leaker",  # a good capsule thrown out costs far less than a leaker shipped
    ),
    "leaker_landed": Question(
        "leaker_landed",
        "Did the capsule just dropped land inside the red leaker bin? The left picture is before the drop, the right after.",
        ("yes", "no"),
        {
            "yes": "the right picture shows one more capsule inside the bin walls than the left",
            "no": "no new capsule inside the bin: it fell outside, or is still in the hand",
        },
        safe_option="no",
    ),
    "tray_clear": Question(
        "tray_clear",
        "Are all the leakers gone from this tray?",
        ("yes", "no"),
        {
            "yes": "no capsule sits in a pool; a pool with no capsule in it is where a leaker was taken out",
            "no": "a capsule still sits in a pool of its fill",
        },
        safe_option="no",
    ),
}


@dataclass
class RungAnswer:
    rung: str
    probs: dict[str, float]
    latency_ms: float
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> str:
        return max(self.probs, key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        return float(max(self.probs.values())) if self.probs else 0.0


@dataclass
class Decision:
    question_id: str
    options: tuple[str, ...]
    state_text: str
    frame_path: str | None
    rungs: list[RungAnswer]
    answer: str
    answered_by: str
    executable: dict[str, bool]
    t_sim: float
    truth: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question_id,
            "options": list(self.options),
            "state_text": self.state_text,
            "frame": self.frame_path,
            "rungs": [
                {"rung": r.rung, "probs": r.probs, "latency_ms": round(r.latency_ms, 1), "error": r.error, **({"extra": r.extra} if r.extra else {})}
                for r in self.rungs
            ],
            "answer": self.answer,
            "answered_by": self.answered_by,
            "executable": self.executable,
            "t_sim": round(self.t_sim, 2),
            "truth": self.truth,
            "context": self.context,
        }


class Rung(Protocol):
    name: str

    def answer(self, question: Question, state_text: str, image_b64: str | None) -> RungAnswer: ...


def uniform(options: tuple[str, ...]) -> dict[str, float]:
    return {o: 1.0 / len(options) for o in options}


def normalised(probs: Mapping[str, float], options: tuple[str, ...]) -> dict[str, float]:
    vals = {o: max(float(probs.get(o, 0.0)), 0.0) for o in options}
    total = sum(vals.values())
    return {o: v / total for o, v in vals.items()} if total > 0 else uniform(options)


def timed(name: str, options: tuple[str, ...], fn: Callable[[], tuple[dict[str, float], dict[str, Any]]]) -> RungAnswer:
    """Run a rung's call, time it, and never let an exception escape (a failed rung is a logged error)."""
    t0 = time.perf_counter()
    try:
        probs, extra = fn()
        return RungAnswer(name, normalised(probs, options), (time.perf_counter() - t0) * 1000.0, None, extra)
    except Exception as error:  # noqa: BLE001 - a rung failing must escalate, not crash the robot
        return RungAnswer(name, uniform(options), (time.perf_counter() - t0) * 1000.0, f"{type(error).__name__}: {error}"[:200])


@dataclass
class Step:
    """One rung of the ladder with the confidence it must reach to be trusted: one number, or one per question id."""

    rung: Rung
    threshold: float | Mapping[str, float]
    questions: frozenset[str] | None = None  # only these question ids reach this rung (a tray scale answers in_bin only)

    def covers(self, question_id: str) -> bool:
        return self.questions is None or question_id in self.questions

    def threshold_for(self, question_id: str) -> float:
        if isinstance(self.threshold, Mapping):
            return float(self.threshold.get(question_id, 1.0))
        return float(self.threshold)


ExecutableFn = Callable[[Question], dict[str, bool]]
RuleFn = Callable[[Question], str | None]


class Ladder:
    """Climb the rungs until one is confident enough and its answer can be executed."""

    def __init__(self, steps: list[Step], rules: RuleFn | None = None) -> None:
        self.steps = steps
        self.rules = rules

    def decide(
        self,
        question: Question,
        state_text: str,
        image_b64: str | None,
        executable: ExecutableFn,
        *,
        t_sim: float = 0.0,
        frame_path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Decision:
        can = executable(question)
        answers: list[RungAnswer] = []
        forced = self.rules(question) if self.rules is not None else None
        if forced is not None:
            answers.append(RungAnswer("rule", {o: (1.0 if o == forced else 0.0) for o in question.options}, 0.0))
            return Decision(question.id, question.options, state_text, frame_path, answers, forced, "rule", can, t_sim, context=context or {})
        for step in self.steps:
            if not step.covers(question.id):
                continue
            ans = step.rung.answer(question, state_text, image_b64)
            answers.append(ans)
            if ans.error is not None:
                continue
            best = max((o for o in question.options if can.get(o, True)), key=lambda o: ans.probs[o], default=None)
            if best is not None and ans.probs[best] >= step.threshold_for(question.id):
                return Decision(question.id, question.options, state_text, frame_path, answers, best, step.rung.name, can, t_sim, context=context or {})
        fallback = question.safe_option or (answers[-1].top if answers else question.options[0])
        return Decision(question.id, question.options, state_text, frame_path, answers, fallback, "safe_option", can, t_sim, context=context or {})
