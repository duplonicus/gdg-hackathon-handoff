"""Handoff: a behaviour tree that stops to ask small decision models, on the MH3 in Genesis.

Layout of this package (stage 1, the foundation):

- ``sim/``      the simulator interface (``base.py``), its Genesis implementation
                (``genesis_sim.py``) and a kinematic fake for tests (``fake.py``).
                Nothing outside ``sim/genesis_sim.py`` imports Genesis or the kit.
- ``scene.py``  the table, the parts and the bins as plain numbers, plus the per-seed jitter.
- ``checks.py`` success checks that are plain functions of a world-state snapshot.
                They decide a trial, and later label training data and serve as rewards.
- ``skills.py`` the robot's skills (drive, look, reach, grasp, lift, place, home).
                Each is a callable behind a name, so a learned policy can replace one.
- ``tasks/``    sequences of skills: ``pick_place.py`` is the stage 1 task.

The kit's own package ``mh3_genesis`` is left untouched.
"""
