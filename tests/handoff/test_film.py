"""The film recorder (Nov 7): one frame per 1/fps of sim time, each mapped to its sim and wall time."""

from __future__ import annotations

import json

import numpy as np
import pytest

from handoff.film import SORT_ANGLES, FilmRecorder, camera_specs


class FakeSim:
    def __init__(self) -> None:
        self.time = 0.0

    def step(self, n: int = 1) -> None:
        self.time += 0.01 * n

    def capture(self, name: str):
        class Frame:
            rgb = np.zeros((4, 6, 3), dtype=np.uint8)

        return Frame()


def test_frames_follow_sim_time_and_stop_cleanly(tmp_path):
    sim = FakeSim()
    rec = FilmRecorder(sim, tmp_path, ["front", "wide"], fps=10)
    rec.start()
    for _ in range(105):  # 1.05 s of sim at 0.01 s a step
        sim.step()
    rec.stop()
    sim.step(50)  # after stop: no more frames
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert rec.frames == len(rows) == 11  # t = 0.0, 0.1, ..., 1.0
    assert sorted(p.name for p in (tmp_path / "front").iterdir())[-1] == "f00010.jpg"
    assert len(list((tmp_path / "wide").iterdir())) == 11
    assert all(r["sim_t"] >= r["frame"] / 10 - 1e-9 for r in rows)


def test_a_long_step_owes_several_frames(tmp_path):
    sim = FakeSim()
    rec = FilmRecorder(sim, tmp_path, ["front"], fps=30)
    rec.start()
    sim.step(21)  # 0.21 s in one step
    rec.stop()
    assert rec.frames == 7  # t 0 to 0.2 at 1/30 s: frames for 0, 1/30, ..., 6/30


def test_unknown_angle_is_refused():
    with pytest.raises(ValueError):
        camera_specs(SORT_ANGLES, ["front", "nope"], (640, 360))


def test_the_page_shown_is_the_latest_one_at_or_before_the_frame():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("compose_operator_shot", root / "scripts" / "compose_operator_shot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    times = [10.0, 12.0, 15.0]
    assert mod.page_for(times, 9.9) is None
    assert mod.page_for(times, 10.0) == 0
    assert mod.page_for(times, 14.99) == 1
    assert mod.page_for(times, 99.0) == 2


def test_page_states_follow_the_answerer_log():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("render_operator_page", root / "scripts" / "render_operator_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [
        {"wall_t": 100.0, "event": "page_up"},
        {"wall_t": 110.0, "event": "question", "id": 0},
        {"wall_t": 114.0, "event": "answered", "id": 0, "answer": "defective"},
        {"wall_t": 130.0, "event": "question", "id": 1},
        {"wall_t": 250.0, "event": "gone", "id": 1},
    ]
    got = [(s["state"], s["wall_t"], s["id"]) for s in mod.states(rows)]
    assert got == [("idle", 100.0, None), ("question", 110.0, 0), ("press", 113.6, 0), ("idle", 114.3, None),
                   ("question", 130.0, 1), ("idle", 250.0, None)]
    # a 0 s plan answers in the same pass: the press still comes after its question
    quick = mod.states([{"wall_t": 10.0, "event": "question", "id": 0}, {"wall_t": 10.01, "event": "answered", "id": 0, "answer": "no"}])
    assert [(x["state"], x["id"]) for x in quick] == [("idle", None), ("question", 0), ("press", 0), ("idle", None)]
    assert quick[2]["wall_t"] > quick[1]["wall_t"]
    assert "Robot operator" in mod.page_html()


def test_a_second_take_into_the_same_folder_is_refused(tmp_path):
    sim = FakeSim()
    rec = FilmRecorder(sim, tmp_path, ["front"], fps=10)
    rec.start()
    sim.step(5)
    rec.stop()
    with pytest.raises(FileExistsError):
        FilmRecorder(FakeSim(), tmp_path, ["front"], fps=10).start()


def test_an_empty_angle_folder_is_not_a_take(tmp_path):
    (tmp_path / "front").mkdir()
    assert not FilmRecorder.holds_take(tmp_path, ["front"])
    (tmp_path / "front" / "f00000.jpg").write_bytes(b"x")
    assert FilmRecorder.holds_take(tmp_path, ["front"])


def test_a_question_that_would_flash_by_is_held_readable():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("compose_operator_shot", root / "scripts" / "compose_operator_shot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frames = [{"frame": i, "wall_t": 100.0 + i} for i in range(10)]
    shots = [{"wall_t": 100.0, "state": "idle"}, {"wall_t": 104.0, "state": "question"}, {"wall_t": 105.0, "state": "press"}, {"wall_t": 106.0, "state": "idle"}]
    seq = mod.plan(frames, shots, fps=10, hold_question=2.0, hold_press=0.5)
    assert [k for _, k in seq].count(1) == 20  # the 1-frame question held to 2 s at 10 fps
    assert [k for _, k in seq].count(2) == 5
    assert [src for src, k in seq if k == 1] == [4] * 20  # held on the question's first frame
    assert [src for src, _ in seq][-4:] == [6, 7, 8, 9]  # the rest plays on in order
    long_q = mod.plan(frames, [{"wall_t": 100.0, "state": "question"}], fps=1, hold_question=2.0, hold_press=0.5)
    assert len(long_q) == 10  # a question already shown long enough is not padded
