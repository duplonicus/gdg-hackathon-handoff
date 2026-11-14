# handoff

The MH3 sorts parts into bins. A behaviour tree runs fixed logic; at marked points it stops and asks a ladder of decision models (stage 2). This package is the foundation: the simulator interface, the scene, the skills and the checks.

The kit's own package `mh3_genesis` is never modified. This package sits beside it.

## Run it

```powershell
.\run.ps1 pytest tests/handoff -q                             # the CPU tests (no GPU, no models)
.\run.ps1 scripts/run_trials.py --trials 10 --grasp both      # stage 1: 10 jittered pick-and-place trials, fingers and weld
.\run.ps1 scripts/run_sort.py --episodes 24 --ladder oracle --fault cycle --frames   # stage 2: log decisions
.\run.ps1 scripts/benchmark.py describe --log output/sort/<run>                      # Gemma descriptions for text-only arms
.\run.ps1 scripts/benchmark.py replay  --log output/sort/<run> --arm gemma-e4b       # one arm at a time
.\run.ps1 scripts/benchmark.py score   --log output/sort/<run>                       # the numbers (a --scales log also gets a scale arm, from its logged readings)
.\run.ps1 scripts/demo.py                                      # watch it: viewer, kit panel, operator page, the frozen ladder (d3 27B)
```

These are the Windows commands. On Linux or macOS use `bash run.sh` in place of `.\run.ps1`. Always go through the launcher: a plain `uv run` on Windows swaps the CUDA build of torch for the CPU one.

`demo.py` runs the frozen ladder by default: d3 27B (llama-server on `127.0.0.1:8774`, `D:\Toolkits\build\serve_d3-27b.cmd`, the docs/PREREG.md flags) answers alone at 0.80 confidence or more, anything less goes to the operator page. Before it starts d3 (`servers.py`) it stops the Decider 2B, unloads Ollama's models, and refuses when Windows commit would pass 110 GB or the GPU has under about 19 GB free. d3 is started hidden, its output in `serve_d3-27b.log` beside the script, and is left running afterwards. `--scales` puts the tray load cells ahead of d3 on in_bin; that rung is not part of the confirmed ladder.

`--ladder local` is the Oct 11 ladder: the Strands Decider 2B on `127.0.0.1:8765` (native Windows, `D:\Models\decision-servers\decider\serve.cmd`, weights from `D:\Models\hf-win`, offline), then Gemma `gemma4:e4b` on Ollama `127.0.0.1:11434` (asked only on in_bin, the one question its thresholds let it answer), then the operator. It stops d3 first. On Linux the Decider is a transient systemd user unit named `handoff-decider`:

```bash
systemd-run --user --unit=handoff-decider --collect bash -lc '~/dev/decider-server/.venv/bin/strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8765 --vision'
journalctl --user -u handoff-decider -f      # its log
systemctl --user stop handoff-decider        # free the 4.7 GB of GPU it holds
```

`HANDOFF_DECIDER_START` replaces that start command with any shell line (for a Decider served from somewhere else), `HANDOFF_DECIDER_URL` the address.

Trial results land in `output/trials/<run>/` (one JSON line per trial plus frames). Sort runs land in `output/sort/<run>/`: `decisions.jsonl` (one line per decision: question, options, state text, frame path, every rung's probabilities and latency, the answer, who answered, and the truth), `frames/`, `episodes.jsonl`, `load_cells.log` (each tray's load at every in_bin check, plus the scales' verdict under `--scales`) and a readable tree trace per episode. The demo writes `run.log` (everything shown on the panel and the viewer) to `output/demo/<run>/`.

## What is where

| File | What it does |
|---|---|
| `sim/base.py` | The `Simulator` interface. Everything above it talks to the robot only through this. |
| `sim/genesis_sim.py` | The Genesis implementation, built on the kit's `MH3Simulation`. The only file that imports Genesis. |
| `sim/fake.py` | A kinematic fake for tests: teleports instead of simulating. |
| `config/sensors.yaml` | The robot's own sensors, each with a cited part spec in `sim/base.py`: IMUs (ZED X IMU) on the left hand, head, torso and base; the left arm's joint torques; a tactile skin on the left palm and phalanges (FlexiForce A201); a force sensor on each left fingertip (FSR 400). Read with `read_imus`, `read_joint_torques`, `read_tactile`, `read_fingertips`; every one has its own panel window in the demo. |
| `bump.py`, `config/bump.yaml` | Bump detection from the robot's own sensors: the hand IMU decides by default (joint torque by momentum observer and the skin also monitor); the box it names is an estimate. The contact solver is only the reference (`contact_truth`). Placement test: `scripts/probe_bump_sensors.py` then `scripts/score_bump_sensors.py`. |
| `scene.py` | Table, parts and bins as numbers, plus per-seed jitter of the parts and (with `bin_jitter_m`) of the bins. |
| `perception.py` | What the robot works out from camera data instead of being told: `locate_bins` finds each tray in the bins camera's colour image and point cloud. |
| `sensors.py` | A simulated load cell under each tray (true force from the simulator, plus a written-down noise model), the rule that turns the readings into an in_bin answer, and the scale rung that puts it first on the in_bin ladder (`run_sort.py --scales`). |
| `checks.py` | Success checks as plain functions of a state snapshot (`grasp_held`, `in_bin`, `on_floor`, ...). |
| `skills.py` | `drive_to_table`, `look_at_table`, `reach`, `approach`, `grasp`, `lift`, `move_to_bin`, `lower` (down into the bin to 4 cm above the wall top, as far as the 19 cm hand allows over a 16 cm bin), `release`, `go_home`, `retract`, plus `look_at` (the head follows the work) and `open_hand` (lets go over a second; snapping open flicked parts out of the bin). Each skill is registered by name so a learned policy can replace it. |
| `tasks/pick_place.py` | The stage 1 sequence and what counts as a successful trial. |
| `finish.py` | How parts are drawn: good, or one of two defects (light lines, grey patches) with a severity dial, as textures on a UV cube mesh. The robot is asked only good or defective. |
| `decider.py` | Typed questions with fixed option lists, the ladder (rungs with thresholds, executable check, safety rules). |
| `rungs.py` | What a ladder can ask: a Decider server, Gemma by first-token logprobs, Gemma by self-reported confidence, an oracle, a replay. |
| `operator.py` | The human rung: a small browser page; the robot holds while it waits. Buttons, or a spoken answer the person confirms. |
| `voice.py` | Spoken answers: Whistle (`scripts/get_whistle.ps1`) turns audio into text, the ladder's model maps the text to an option. |
| `logbook.py` | Every decision as one JSON line with its frame. |
| `bt.py` | A minimal behaviour tree (Sequence, Selector, Retry, ForEach, Action, Condition) with a readable trace. |
| `tasks/sort.py` | The stage 2 task: the tree that finds the bins, then feeds, inspects, asks, picks, places, verifies and recovers (and dances, with `--dance`). |
| `benchmark.py` | Scoring over logged decisions: right-decision rate, confidently-wrong rate, calibration, latency, human-call rate; offline ladders and measured thresholds. |

## What the robot is told and what it finds out

- **Finds out:** where the bins are. Each episode moves both trays up to 2 cm (`run_sort.py --bin-jitter`, default 0.02; the stage 1 trials keep them fixed). The sort's first step locates them from the fixed camera over the bins and the arm aims at what it saw. If a tray is not wholly in view the sort stops and says so. Ground truth uses where the trays really are; both positions are logged.
- **Still told:** the pick station's position, the part's exact pose for the grasp, and the inputs to the safe-to-grasp rule.
- **Return to home between parts** is on by default. `run_sort.py --no-return-home` lifts clear and starts back without waiting: about 6% faster, 19 of 20 against 20 of 20 (see `VALIDATION.md`). STALE since Oct 21: measured before go-home started crossing to a spot clear of both trays at the clearance height; not re-measured.

## Terms

- **Pose**: position in metres plus orientation as a quaternion (w, x, y, z).
- **Wrist**: the hand's root link (`left_gripper_base`). Arm targets are wrist poses.
- **IK**: inverse kinematics, solving joint angles that put the wrist at a wanted pose. The joints are then driven there by PD position controllers (a spring and a damper per joint), so moves take simulated time.
- **Settle**: step physics until things stop moving.
- **Weld**: a runtime constraint that pins the part to the hand. The touch grip uses it as a stand-in for a rigid part's friction grip (below); `--grasp weld` on `run_sort.py` and `run_trials.py` adds it under any grip, as the comparison arm for the finger grasp.

## The grasp, and why the part is 4 cm

The MH3 hand has 5 cm fingers on a 7 cm thick palm. A fully curled fingertip reaches only 1 cm past the palm face, so the hand cannot wrap around a block. Its grasp is a pinch: the thumb tip and finger pads press the part against the palm, and friction does the rest.

Three things made that pinch reliable (measured Oct 11, 10 jittered trials each):

1. **Soft fingers.** The kit gives finger joints the body's stiffness (350) and no effort limit, so a closed hand pressed with about 200 N and jittered the part out. Stiffness 5 holds without crushing.
2. **A 4 cm part.** A 5 cm block only ever caught on the fingertips (3 to 4 of 10 at any stiffness); 4 cm sits in the pinch.
3. **Yaw alignment.** The hand turns to match the part's heading (the wrist joints allow about 17 degrees, jitter is 15). Without it the most-yawed seeds drop the part during the carry.

Wrist moves are eased over time because a one-step target made the arm whip and drop what it held, and follow the straight line to the target (IK every centimetre, each point seeded with the last). Interpolating the joints from start to goal instead swung the hand on an arc that sagged 8 cm mid-crossing onto the tray walls (Oct 21). Crossings hold the hand level but let it turn about vertical (`free_yaw`): holding the grasp yaw along a line drove the 17-degree wrist joints into their stops. After one failed robot re-pick of a part the operator places it.

**The fixed closure sinks in.** Closing to 0.9 commands a finger position inside the part, so the fingers drive into it until the contact solver pushes back: 13 to 16 mm of penetration on a Genesis CPU check (Oct 21, `scripts/probe_grip_cpu.py`), the clipping the demo showed before Oct 27 (still there with `--grip fixed` or `--close`). A force-controlled grasp (`grasp_control="force"`, `--grip force` on the demo and runners) closes until the fingertip sensors feel the grip force and a grip keeper holds it through the carry; it sinks about 1 mm but does not hold on the lift (a fingertip sensor cannot tell the part from the hand's own thumb), so it is not the default anywhere. Closing to 0.75 instead (`--close 0.75`) sinks about 6 mm and held on all 6 seeds tried for the lift and a 1 s hold, **but drops the part on the carry**: the thumb never touches it (0 N), so it hangs from the index finger alone (the Oct 21 demo and `scripts/probe_carry.py`). 0.85 held every carry tried. **The touch grip** (`--grip touch`) stops the fingers at the surface instead: it closes until the fingertip sensors feel the part, rests the fingers there and welds the part to the hand, a simulation stand-in for a rigid part's friction grip (4 mm of sink at most against 16 at 0.9, 27 of 30 carries clean against 20 on seeds 1 to 10). **The demo defaults to the touch grip** (since Oct 27; `--close` alone still means the fixed closure); `run_sort.py` keeps the fixed closure as its default so the PREREG commands rerun as recorded. The robot dances only with `--dance`, after backing 0.8 m off the table. Findings: `VALIDATION.md`.

## The four questions

| Question | Options | Kind | Camera |
|---|---|---|---|
| finish | good, defective (three answers until Oct 12) | judgement | inspection camera over the pick station |
| safe_to_grasp | grasp, re_look, ask_operator, abort | safety (also a hard rule from sim state) | close-up at the station |
| in_bin | yes, no | judgement | camera over the bins; with `--scales`, the tray load cells answer first and the camera only when they cannot tell |
| sort_complete | yes, no | safety (also a hard rule) | camera over the bins |

Every rung sees the same option list. Before an answer is acted on it is checked for executability (grasp needs an open hand and the part in the station). Safety questions get a hard rule as well: a part on the floor or a neighbour within 6 cm forces abort; a part on the floor forces "not complete".

## Benchmark (test03: 90 episodes, 974 decisions, Oct 13 to 14)

Every arm answered the same recorded decisions (frames and state text from `output/sort/test03`, made with the current robot: bins moved per episode and located by the robot, gradual release). Thresholds are chosen per rung and per question on 27 held-out episodes; the other 676 decisions are scored. "Confidently wrong" means answered at confidence 0.9 or more and wrong. Every rate carries a 95 percent by-episode bootstrap interval, and `score` also reports the "judgement" decisions on their own (everything but safe_to_grasp, which a hard rule decides and where every arm sits at the base rate). Each arm row records the host it ran on; latencies from different hosts are not comparable. Full tables: `VALIDATION.md`; charts: `.\run.ps1 scripts/charts.py --log output/sort/test03`.

| arm | right [95%] | confidently wrong [95%] | calibration error | mean confidence | ms (host) |
|---|---|---|---|---|---|
| pplx-decider 27B Artemis, frame | 0.805 [0.76, 0.85] | 0.016 [0.00, 0.03] | 0.068 | 0.87 | 572 (Windows Ollama) |
| Decider 2B, frame | 0.780 [0.73, 0.83] | **0.000** [0.00, 0.00] | **0.078** | 0.72 | 325 (WSL unit) |
<!-- STALE (Oct 16): this headline is one split (seed 0) and leaves out the acted-wrong rate. Ten-split figures and the person-cost curve: VALIDATION.md, "Headline ladder over ten splits". Rescore after the evals and the test04 confirmation. -->
| Gemma 4 e4b (first-token logprobs) | 0.768 [0.75, 0.79] | 0.195 [0.17, 0.22] | 0.191 | 0.95 | 162 (WSL Docker) |
| Gemma 4 e4b, own confidence | 0.720 [0.70, 0.75] | 0.280 [0.25, 0.30] | 0.264 | 0.98 | 374 (WSL Docker) |
| Decider E4B, text + Gemma description | 0.562 [0.53, 0.60] | 0.034 [0.02, 0.05] | 0.253 | 0.80 | 184 (WSL unit) |
| Decider 2B, text + Gemma description | 0.547 [0.52, 0.58] | 0.003 [0.00, 0.01] | 0.262 | 0.81 | 115 (WSL unit) |
| Decider E2B, text + Gemma description | 0.543 [0.51, 0.57] | 0.028 [0.02, 0.04] | 0.324 | 0.71 | 150 (WSL unit) |
| Gemma 4 e2b, own confidence | 0.470 [0.45, 0.49] | 0.530 [0.51, 0.55] | 0.478 | 0.95 | 357 (Windows Ollama) |
| Gemma 4 e2b | 0.464 [0.44, 0.49] | 0.522 [0.49, 0.55] | 0.528 | 0.99 | 205 (WSL Docker) |
| Gemma 4 12b | 0.359 [0.31, 0.41] | 0.407 [0.36, 0.45] | 0.533 | 0.89 | 388 (Windows Ollama) |

Ladders come in three families. A: decision models reading the frame, no vision-language model anywhere. B: decision models reading the state text plus Gemma e4b's description of the frame. C: Gemma as a rung. "Decided alone and right" counts the decisions the models answered without the person and got right, out of all 676.

| ladder (thresholds per rung and question) | decided alone and right | handed to a person | confidently wrong |
|---|---|---|---|
| A: Decider 2B, then pplx-decider 27B, then a person | 0.793 [0.75, 0.83] | 0.13 [0.11, 0.14] | 0.006 [0.00, 0.01] |
| C: Decider 2B, then Gemma e4b, then a person | 0.786 [0.75, 0.82] | 0.13 [0.11, 0.15] | 0.009 [0.00, 0.02] |
| A: pplx-decider 27B, then a person | 0.645 [0.62, 0.67] | 0.32 [0.29, 0.34] | 0.007 [0.00, 0.02] |
| B: Decider 2B frame, then E4B text, then a person | 0.620 [0.59, 0.65] | 0.28 [0.26, 0.31] | 0.001 [0.00, 0.00] |
| A: Decider 2B, then a person | 0.571 [0.54, 0.60] | 0.36 [0.34, 0.38] | 0.000 [0.00, 0.00] |
| C: Gemma e4b, then a person | 0.521 [0.49, 0.55] | 0.41 [0.40, 0.41] | 0.068 [0.04, 0.09] |
| B: Decider 2B text, then E4B text, then a person | 0.178 [0.15, 0.20] | 0.79 [0.76, 0.82] | 0.004 [0.00, 0.01] |
| a person answers everything | 0.00 | 1.00 | 0.000 |

What the numbers say:

- **Calibration is the gap, not accuracy.** The two decision models that read the frame are the only calibrated arms. The 2B is never confidently wrong on 676 decisions; Gemma e4b is right about as often and confidently wrong one time in five.
- **A ladder without a VLM ties the best ladder with one.** Decider 2B then the 27B decides 79 percent alone and right and hands off 13 percent, the same as Decider 2B then Gemma, with fewer confidently-wrong answers. The second rung earns its place on in_bin, where the 2B alone hands off 79 percent; on the decisions the 2B escalates, the 27B is right 76 percent of the time against the 2B's own guess at 58.
- **Nobody earned trust on sort_complete** (91 percent handed to the person in every ladder) **or on safe_to_grasp**, where the hard rule from sim state does the work.
- **Text-only Deciders cannot do the job alone** (0.16 to 0.18 decided alone): the state text does not describe the part. That is prompt design, not the models.
- **Self-stated confidence is the cautionary row.** The Gemma self-confidence ladder never hands off and is confidently wrong a third of the time.
- **Prompt matters as much as model.** Gemma 12b answers the cautious option on nearly everything at probability near 1.0 while describing the same frames correctly in prose; the Oct 11 bin-colour prompt change took e4b on in_bin from 0.14 to 1.00.

Still to run: the live-ladder check (the chosen ladder for real on 20 fresh episodes, live against replay), cloud arms, and the open models not yet pulled. Stage 3 fine-tunes the Decider 2B on these logged decisions and their sim-labelled truth.

## Checks are the contract

`checks.py` never looks at the simulator. It reads a `WorldState` snapshot, so the same functions decide a trial today, label logged decisions with what really happened in stage 3, and serve as rewards in stage 4.
