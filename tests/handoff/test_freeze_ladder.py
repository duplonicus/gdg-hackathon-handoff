"""freeze_ladder.py confirm refuses a frozen ladder whose rungs answered only part of the confirmation log (Nov 7)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("freeze_ladder", ROOT / "scripts" / "freeze_ladder.py")
freeze_ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_ladder)


def test_a_fully_replayed_ladder_is_not_short():
    labelled = set(range(100))
    arms = {"a": {i: {} for i in range(100)}, "b": {i: {} for i in range(97)}}
    assert freeze_ladder.short_rungs(["a", "b"], arms, labelled) == []


def test_a_rung_stopped_part_way_is_short():
    labelled = set(range(100))
    arms = {"a": {i: {} for i in range(100)}, "b": {i: {} for i in range(40)}}
    assert freeze_ladder.short_rungs(["a", "b"], arms, labelled) == [("b", 40)]


def test_answers_to_unlabelled_decisions_do_not_count():
    labelled = set(range(100))
    arms = {"a": {i: {} for i in range(50, 200)}}
    assert freeze_ladder.short_rungs(["a"], arms, labelled) == [("a", 50)]
