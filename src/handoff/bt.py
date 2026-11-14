"""A minimal behaviour tree: enough to run the sort with fixed logic and visible structure.

A behaviour tree is a tree of nodes ticked from the root. A **Sequence** runs its children in
order and fails at the first failure; a **Selector** tries children in order and succeeds at the
first success (that is how fallbacks and recoveries are expressed); a **Retry** re-runs its child
up to N times. Leaves are **Action** (do something, report success or failure) and
**Condition** (check something). Every tick is recorded, so the trace of a run is the tree's own
log, which is what an operator reads to see where the robot was and why.

py_trees is the organisers' suggestion; this file is about 80 lines so the whole control
structure stays readable and dependency-free. Skills stay blocking, so there is no RUNNING
status: a tick returns when the leaf is done.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


Blackboard = dict[str, Any]


@dataclass
class Trace:
    """One line per tick: depth, node name, result. The run's readable story."""

    lines: list[tuple[int, str, str]] = field(default_factory=list)

    def add(self, depth: int, name: str, status: Status | str) -> None:
        self.lines.append((depth, name, status.value if isinstance(status, Status) else status))

    def text(self) -> str:
        return "\n".join(f"{'  ' * d}{n}: {s}" for d, n, s in self.lines)


class Node:
    name: str = "node"

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        raise NotImplementedError


class Action(Node):
    def __init__(self, name: str, fn: Callable[[Blackboard], bool | Status]) -> None:
        self.name, self.fn = name, fn

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        result = self.fn(bb)
        status = result if isinstance(result, Status) else (Status.SUCCESS if result else Status.FAILURE)
        note = bb.pop("note", None)  # an action may leave a one-line reason for the trace
        trace.add(depth, self.name, f"{status.value}: {note}" if note else status)
        return status


class Condition(Action):
    """Same as an Action, named so the tree reads as a check rather than a doing."""


class Sequence(Node):
    def __init__(self, name: str, *children: Node) -> None:
        self.name, self.children = name, list(children)

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        trace.add(depth, self.name, "start")
        for child in self.children:
            if child.tick(bb, trace, depth + 1) is Status.FAILURE:
                trace.add(depth, self.name, Status.FAILURE)
                return Status.FAILURE
        trace.add(depth, self.name, Status.SUCCESS)
        return Status.SUCCESS


class Selector(Node):
    def __init__(self, name: str, *children: Node) -> None:
        self.name, self.children = name, list(children)

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        trace.add(depth, self.name, "start")
        for child in self.children:
            if child.tick(bb, trace, depth + 1) is Status.SUCCESS:
                trace.add(depth, self.name, Status.SUCCESS)
                return Status.SUCCESS
        trace.add(depth, self.name, Status.FAILURE)
        return Status.FAILURE


class Retry(Node):
    def __init__(self, name: str, child: Node, attempts: int) -> None:
        self.name, self.child, self.attempts = name, child, attempts

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        for attempt in range(1, self.attempts + 1):
            trace.add(depth, f"{self.name} (attempt {attempt}/{self.attempts})", "start")
            if self.child.tick(bb, trace, depth + 1) is Status.SUCCESS:
                trace.add(depth, self.name, Status.SUCCESS)
                return Status.SUCCESS
        trace.add(depth, self.name, Status.FAILURE)
        return Status.FAILURE


class ForEach(Node):
    """Run the child once per item in ``bb[key]`` (setting ``bb[item_key]`` first).

    With ``fail_fast=False`` (the default) every item gets its turn and the node succeeds, noting
    the failed items in ``bb[key + "_failed"]``; the caller decides what a failed item means.
    """

    def __init__(self, name: str, key: str, item_key: str, child: Node, *, fail_fast: bool = False) -> None:
        self.name, self.key, self.item_key, self.child, self.fail_fast = name, key, item_key, child, fail_fast

    def tick(self, bb: Blackboard, trace: Trace, depth: int = 0) -> Status:
        trace.add(depth, self.name, "start")
        failed = []
        for item in list(bb[self.key]):
            bb[self.item_key] = item
            trace.add(depth + 1, f"{self.name}[{item}]", "start")
            if self.child.tick(bb, trace, depth + 2) is Status.FAILURE:
                failed.append(item)
        bb[self.key + "_failed"] = failed
        status = Status.FAILURE if (failed and self.fail_fast) else Status.SUCCESS
        trace.add(depth, self.name, f"{status.value} ({len(failed)} failed)" if failed else status)
        return status
