"""The full skill sequence against the kinematic fake (no GPU)."""

from handoff.scene import Layout, make_episode, make_scene, place_episode
from handoff.sim.fake import FakeSimulator
from handoff.skills import SKILLS, TRAVEL_POSE, TaskContext
from handoff.tasks.pick_place import SEQUENCE, run_pick_place


def make_sim(seed: int = 0) -> tuple[FakeSimulator, Layout, TaskContext]:
    layout = Layout()
    sim = FakeSimulator()
    sim.build(make_scene(layout))
    episode = make_episode(layout, seed)
    place_episode(sim, layout, episode)
    sim.set_joints(TRAVEL_POSE)
    return sim, layout, TaskContext(layout=layout, target_part=episode.parts[0], parts=episode.parts)


def test_every_sequence_step_is_a_registered_skill():
    assert all(name in SKILLS for name in SEQUENCE)


def test_pick_place_succeeds_on_the_fake_for_several_seeds():
    events: list[str] = []
    for seed in range(5):
        sim, layout, ctx = make_sim(seed)
        ctx.on_event = events.append
        result = run_pick_place(sim, ctx)
        assert result.success, (seed, result.failed_at, [s.reason for s in result.steps if not s.ok])
        assert result.checks["in_target_bin"] and not result.checks["in_other_bin"] and not result.checks["on_floor"]
        assert [s.skill for s in result.steps] == list(SEQUENCE)
    assert "pregrasp" in events and "released" in events


def test_weld_mode_attaches_only_after_contact_and_detaches_on_release():
    sim, layout, ctx = make_sim(1)
    ctx.grasp_mode = "weld"
    result = run_pick_place(sim, ctx)
    assert result.success
    part = ctx.target_part
    attach_idx = sim.calls.index(f"attach:{part}")
    # the force grasp ramps the hand closed (skills.grasp_force), so the last closing command, whatever its value
    close_idx = max(i for i, c in enumerate(sim.calls[:attach_idx]) if c.startswith("set_hand:left:") and float(c.rsplit(":", 1)[1]) > 0.5)
    assert close_idx < attach_idx < sim.calls.index(f"detach:{part}")


def test_out_of_reach_part_fails_at_reach_with_a_reason():
    sim, layout, ctx = make_sim(0)
    sim.set_object_pose(ctx.target_part, (1.0, 0.25, layout.part_center_z))
    result = run_pick_place(sim, ctx)
    assert not result.success
    assert result.failed_at == "reach"
    assert "missed" in result.steps[-1].reason


def test_rework_bin_target_works_too():
    sim, layout, ctx = make_sim(2)
    ctx.target_bin = "rework"
    result = run_pick_place(sim, ctx)
    assert result.success
    assert result.checks["in_target_bin"]


def test_drive_parks_near_the_origin():
    sim, layout, ctx = make_sim(0)
    result = SKILLS["drive_to_table"](sim, ctx)
    assert result.ok, result.data
    assert abs(sim.base_pose().pos[0] - layout.park_x) <= 0.05


def test_yaw_alignment_turns_the_hand_and_the_offset_with_the_part():
    import numpy as np

    from handoff.skills import HOME_QUAT, grasp_quat, grasp_wrist_pos, part_yaw

    sim, layout, ctx = make_sim(0)
    sim.set_object_pose(ctx.target_part, (0.55, 0.15, layout.part_center_z), 100.0)  # 100 deg folds to 10 deg
    assert abs(part_yaw(sim, ctx) - 10.0) < 1e-6
    quat = grasp_quat(sim, ctx)
    assert not np.allclose(quat, HOME_QUAT["left"])
    pos = grasp_wrist_pos(sim, ctx)
    off = pos - np.array([0.55, 0.15, layout.part_center_z])
    assert abs(np.linalg.norm(off[:2]) - np.linalg.norm(ctx.grasp_offset[:2])) < 1e-9  # same length, rotated
    sim.set_object_pose(ctx.target_part, (0.55, 0.15, layout.part_center_z), 40.0)  # beyond the hand's turn: clamped
    ctx.data.clear()
    assert abs(part_yaw(sim, ctx) - ctx.max_align_deg) < 1e-6
    ctx_fixed = TaskContext(layout=layout, target_part=ctx.target_part, align_yaw=False)
    assert np.allclose(grasp_quat(sim, ctx_fixed), HOME_QUAT["left"])
