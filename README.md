# A Humanoid That Knows When to Ask: Vision Decision Models on the Shop Floor

An entry to the GDG Windsor hackathon **Build with AI: Smart Automation, Robotics, and Vision** (Kaggle), built on the [MH3 Genesis starter](docs/STARTER.md).

A Mirsee MH3 humanoid in the Genesis simulator works two cells. **A behaviour tree runs everything that must never fail** (motion, safety stops, hard rules, sensor checks). **The judgement calls go to a vision model that answers only when it is measurably sure**, and to a person on an operator page when it is not. Everything runs offline on one GPU.

- **Phase 1, validated:** the robot sorts painted parts into a good tray and a rework tray. The decision ladder was frozen before its test and scored once on 300 fresh episodes.
- **Phase 2, built after Phase 1 was validated:** the robot picks leaking soft-gel capsules off an inspection tray. Same architecture, new question: which kind of vision model can a plant trust to act alone, one that writes an answer (a VLM) or one that scores each option (a VDM)? Its confirmation run is still to come.

The full story is in [WRITEUP.md](WRITEUP.md); every number has a row in [VALIDATION.md](VALIDATION.md) with the command that produced it.

## Phase 1: sorting painted parts (validated)

### The result

The ladder (d3 27B, then a person, at 0.80 confidence) was chosen on one test set, frozen in [docs/PREREG.md](docs/PREREG.md), then scored once on 300 fresh episodes (3,262 decisions, 2,262 of them judgement questions: finish, in_bin, sort_complete):

| Judgement questions, test04 | Result (95% interval) | Predicted before the test |
|---|---|---|
| Acted wrong on its own | **2.6%** (1.7 to 3.5) | 1.8% |
| Handed to a person | 18.5% (16.8 to 20.2) | 20.7% |
| Decided alone and right | **78.9%** (77.0 to 80.9) | 77.5% |

**One known blind spot:** from one camera frame, the vision model called every wrong-bin landing "in the bin" (15 landings, checked 29 times, all 29 wrong), because the target tray already held an earlier part each time. A load cell under each tray answers that question instead (`--scales`).

### How a question flows

1. The behaviour tree reaches a question (finish good or defective? safe to grasp? landed in its tray? sort complete?).
2. A hard rule answers it if one applies (crowded station, part on the floor).
3. For "landed in its tray?", with `--scales` on, the tray load cells answer when the reading is clear (not part of the scored ladder).
4. Otherwise d3 27B sees the camera frame and the options. At 0.80 or more on its top option, the robot acts.
5. Below that, the operator page shows the frame, the options and d3's answer. The robot waits. The person clicks an option or answers out loud: Whistle (16.9 MB, on the CPU) transcribes it, d3 proposes the option, and the person confirms it. Speech never acts alone.
6. If the model server stops mid-run, the call fails (at once if refused, after 120 s if it hangs), its rung is skipped and the question goes to the person. The demo will not start without it. If the person does not answer in 120 s, the tree takes the question's safe answer (defective, ask operator, no, no).
7. Every decision is logged with its frame, every answer and confidence, and who decided.

## Phase 2: the capsule cell, VDM against VLM (confirmed)

The MH3 parks once a shift at a capsule inspection cell, base braked: two half-trays of 16 soft-gel capsules (4 by 4) with a leaker bin on each side. A leaker differs from a good capsule only by the pool under it. Fixed cameras watch the tray and each bin; the robot asks which side needs it, whether a block of capsules holds a leaker, which capsule it is, whether the dropped capsule landed in the bin and whether the tray is clear. It picks each leaker with a palm-down pinch and drops it inside that side's bin.

The comparison is the cleanest one in the project: **d3 27B vision, a VDM, is built on Qwen3.8-27B, a VLM.** Same base model, two ways of answering. Intern-Decision-4B is a second VDM; Qwen is read two ways (the probability of its first answer word, and the confidence it states when asked).

**Confirmed on test05**, a log of 150 shifts that no choice had touched, scored once against the protocol frozen beforehand ([docs/PREREG_CAPSULES.md](docs/PREREG_CAPSULES.md)). The ladder picked on the selection log, **Intern-4B > d3 27B > person**, over 3,785 decisions:

- **Acted wrong on 0.2%** (6 decisions; 95% interval 0.0 to 0.3%; predicted 0.5%)
- Handed 2.8% to a person (predicted 2.9%)
- Decided alone and right on 97.1% (predicted 96.5%)
- Of 467 leaker pictures, 389 were caught by a model, 76 went to a person, and 2 were passed as good.

Every arm alone on the same test05 pictures (the leaker calls: 467 leaker pictures, 2,745 clean):

| | Leakers caught | Missed while 0.9+ sure | False alarms |
|---|---|---|---|
| d3 27B (VDM) | 85% | **1** | 36% (all unsure; the ladder hands these on) |
| Intern-Decision-4B (VDM) | 85% | **1** | 0.1% |
| Qwen3.8-27B, stated confidence (VLM) | 63% | **171** | 0.0% |
| Qwen3.8-27B, first word (VLM) | 42% | **98** | 0.0% |

The number that ships a leaker is the middle column: a model that is sure and wrong gets past any threshold. d3 and Qwen share a base model; d3 was trained to answer with a probability for each option, Qwen is read from its words. Intervals and the method are in [VALIDATION.md](VALIDATION.md) ("test05"); the selection logs (sel02, sel03) showed the same pattern. What was settled before the selection log is in [docs/FREEZE.md](docs/FREEZE.md).

## Run it

Windows x64 with an NVIDIA GPU (developed on an RTX 3090, 24 GB). First run the starter's setup (`.\setup.ps1`, see [docs/STARTER.md](docs/STARTER.md)). Always go through `.\run.ps1`.

```powershell
.\run.ps1 pytest tests/handoff -q                  # the tests: fake simulator, no GPU, no models
.\run.ps1 scripts/demo.py --scales --fault wrong_bin # watch it: viewer, panel, operator page at http://127.0.0.1:8790/
.\run.ps1 scripts/demo.py --ladder oracle            # same, with the simulator's truth answering (no model needed)
```

The demo's default ladder needs the d3 27B model served by llama.cpp on `127.0.0.1:8774`. The exact files (with sha256), server build and flags are in [docs/PREREG.md](docs/PREREG.md); `src/handoff/servers.py` starts and stops it through a local `serve_d3-27b.cmd` (its path is in `servers.py`). Without it, use `--ladder oracle`.

Spoken answers on the operator page need Cactus Compute's Whistle speech model and its Windows runner: `.\scripts\get_whistle.ps1` fetches both (about 18.5 MB, pinned commits, checked by sha256) to `D:\Models\whistle`. The runner is called directly, not through the `cactus-needle` Python package, which sends usage counts to Cactus. The model that maps the words to an option is the one the ladder already has loaded (d3 27B on the frozen ladder), so voice adds nothing to the GPU. The read-back ("I heard rework. Please confirm.") is the browser's built-in speech voice. Without Whistle, or with `--ladder oracle`, the page is buttons only.

Making and scoring a decision log:

```powershell
.\run.ps1 scripts/run_sort.py --episodes 24 --ladder oracle --fault cycle --frames  # log decisions (frames and truth always kept; --frames adds event frames)
.\run.ps1 scripts/benchmark.py replay --log output/sort/<run> --arm d3-27b-vision   # replay one model on them
.\run.ps1 scripts/benchmark.py score  --log output/sort/<run>                       # score every arm and ladder
```

[notebooks/](notebooks/) recomputes the test04 numbers from an extract of the decision log, no GPU needed.

The capsule cell (Phase 2):

```powershell
.\run.ps1 scripts/run_capsules.py --episodes 4 --seed0 0 --out output/capsules/smoke                                       # log shifts headless, the simulator's truth answering
.\run.ps1 scripts/run_capsules.py --episodes 1 --config config/capsules_demo.yaml --out output/capsules/watch               # one shift in the viewer
.\run.ps1 scripts/replay_capsules.py --logs output/capsules/smoke --arms d3-27b-vision,qwen-27b-vision                     # replay models on it, one on the GPU at a time
.\run.ps1 scripts/capsule_catch_rates.py output/capsules/smoke                                                            # leakers caught, missed while sure, false alarms
```

## What is where

| Path | What it is |
|---|---|
| `src/handoff/` | The project: behaviour tree, skills, sensors, decision ladder, operator page, benchmark, and the capsule cell (`capsule_*.py`, `tasks/capsules.py`). Its [README](src/handoff/README.md) has the file-by-file map. |
| `src/mh3_genesis/` | The starter kit's package, with small changes (mesh decimation for the viewer, joint labels and run status on the panel, Windows text encoding). |
| `scripts/` | The demo, the sort runner, the benchmark, the ladder pick and freeze, and most of the probes behind VALIDATION's numbers. |
| `tests/handoff/` | The tests (fake simulator; no GPU). |
| `config/` | Robot, scene and bump-detection settings. |
| `docs/PREREG.md` | Phase 1's frozen ladder and scoring protocol, fixed before any model answered test04. |
| `docs/FREEZE.md`, `docs/PREREG_CAPSULES.md` | Phase 2: what is settled before the capsule selection log, and the confirmation protocol, frozen from sel03 before test05. |
| `VALIDATION.md` | Every number, its command, its raw-log path (logs are not in the repo) and what it does not show. |
| `PAPER.md`, `docs/papers/` | Method notes and the papers used. |
| `media/` | Gallery stills and the video's script and slides. |

## Licence

Code: MIT, from the starter (see [LICENSE.md](LICENSE.md)). The MH3 model (`assets/MH3.usd`) is © Mirsee Robotics Inc., licensed only for submissions to this hackathon.
