"""The hand lets go gradually: snapping it open flicked parts over the bin wall."""

from handoff.skills import TaskContext, open_hand


class Hand:
    dt = 0.01

    def __init__(self):
        self.closure, self.targets, self.steps = 0.8, [], 0

    def hand_closure(self, arm):
        return self.closure

    def set_hand(self, arm, closed):
        self.targets.append(closed)

    def step(self, n=1):
        self.steps += n


def test_open_hand_ramps_down_to_fully_open_over_the_release_time():
    sim = Hand()
    open_hand(sim, TaskContext(release_open_s=1.0))
    assert sim.targets[-1] == 0.0
    assert len(sim.targets) >= 10
    assert all(a >= b for a, b in zip(sim.targets, sim.targets[1:]))  # never re-closes on the way
    assert sim.targets[0] < 0.8  # starts from where the hand is, not from fully closed
    assert abs(sim.steps * sim.dt - 1.0) < 0.06
