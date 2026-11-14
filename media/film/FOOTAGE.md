# Film footage (made Nov 7)

Every clip is in-sim cameras at 30 frames per second of SIM time, 1920 x 1080, made by script (media/film/SCRIPT.md, "Footage"). The clips live in `output/film/` (not in git; regenerate with the commands below). What the robot does on screen is what that take's `run.log` says; the numbers to say over a clip are copied from it here.

## P2-A: one capsule shift (beat 11)

- Take: `scripts/film_shift.py --seed 5017 --angles "front,high,wide"`, then `--angles closeup` (same seed, same moves). 69 s of sim.
- Seed 5017: work cell a, 3 leakers (two on one half-tray, one on the other), all 3 binned, no good capsule thrown out, base drift 0 mm.
- Clips: `output/film/p2a/front.mp4` (three-quarter, across the table), `closeup.mp4` (hands, capsules, both bins: the pinch and the drop), `high.mp4` (straight down over the tray), `wide.mp4` (the whole cell and the drive in).
- Seen in the frames: the pools stay on the parchment where leakers were picked; the robot's wheeled base shows under the tabletop in the wide shot (it parks under it, not a clipping bug).

## P1-A and P1-B: the sort cell, an unsure call goes to the page (beats 6 and 8)

- Take: `scripts/demo.py --no-panel --config config/handoff.yaml --film output/film/p1b` (frozen ladder, d3 27B, seed 7, forced drop on the first carry) with `scripts/film_answerer.py --out output/film/p1b --plan "*=model@4"` (the operator agrees with d3, 4 s after each question appears). Run log: `output/demo/20261107-200537/run.log`.
- Result: 3 of 3 parts in the right bin. Four questions went to the page, each answered 4 s after it appeared:
  - in_bin part_15 after the forced drop: d3 "no" 0.70, the person "no".
  - finish part_10: d3 "defective" 0.58, the person "defective" (the clearest unsure call for beat 8).
  - safe_to_grasp part_10: d3 "grasp" 0.78, the person "grasp".
  - safe_to_grasp part_12: d3 "grasp" 0.72, the person "grasp".
- Clips: `output/film/p1b/p1b_front.mp4` (the sim and the operator page side by side, 83 s: each page question is held 2.5 s and each press 0.8 s so they read), and `front.mp4`, `wide.mp4` (P1-A since Nov 11: 53.0 to 69.0 s of `front.mp4`, part_12 from reach to rework bin and arm home, one whole cycle at 1x; it was the drive up, 0 to 8 s, which cut before the grasp), `trays.mp4`.

## P1-D: the model goes down, nobody answers (beat 15)

- Take: `scripts/demo.py --no-panel --config config/handoff.yaml --fault none --film output/film/p1d` with `scripts/film_answerer.py --out output/film/p1d --stop-d3-after 150 --plan "*=truth@4,s1=truth@5,s2=none,s*=truth@3"`. Run log: `output/demo/20261107-204625/run.log`.
- Result: 3 of 3 parts in the right bin. d3 stopped 150 s after the page came up; from then on every question shows "d3-27b-vision ERROR" and goes straight to the page:
  - in_bin part_15: answered "yes" after 5 s.
  - finish part_10: left unanswered; after 120 s the safe answer "defective" was taken (by safe_option).
  - the rest (6 questions) answered as the scene showed, 3 s after each appeared.
- **Label the timeout "120 s, sped up":** the clip runs on sim time, and the robot holds while it waits, so the 120 s wait plays as about 15 s (26.43 s to 41.2 s of `p1d_front.mp4`, from its `composed_index.jsonl`).
- Clips: `output/film/p1d/p1d_front.mp4` (side by side, 94 s, the same holds), `front.mp4`, `wide.mp4`, `trays.mp4`.

## How the operator shots are made

The human-in-the-loop shots are simulated (dup, Nov 6): `film_answerer.py` answers the real operator page on a plan. In P1-D it answers what the simulator says is true (`film_truth`, sent only on `--film` runs and never shown on the page), as a person looking at the scene would. `render_operator_page.py` redraws the real page from what the answerer saved, and `compose_operator_shot.py` lays it beside the sim by wall time. The page's wording for a model that is down ("could not answer (a model is down or failed)") came from this take (27e8183).
