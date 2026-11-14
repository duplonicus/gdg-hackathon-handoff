# Results ledger

One row per number we quote anywhere (the paper notes, the slides, the README, a conversation). Each row says what was measured, the command, where the raw log is, the date, and what the number does NOT show. If a number is not in here, do not quote it.

Raw logs live under `output/`, which git ignores: they exist only on this machine's disk. **The two scored test sets are published**: `output/sort/test04/` and `output/capsules/test05/` (decisions, every model's answers, episodes, the one-time confirmation, test04's behaviour-tree traces; no camera frames) are in the Kaggle dataset https://www.kaggle.com/datasets/duplonicus/humanoid-decision-ladder-logs, each file prefixed `test04_raw_` or `test05_raw_`.

Unless a row says otherwise: Genesis 1.4.3, native Windows, RTX 3090, `--ladder oracle` (the simulator's truth answers every question, so these rows measure the robot's motion and the tree, not the models).

## Sorting reliability and timing (Oct 12, seeds 200 to 219, no forced fault)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Clean sorts fully succeeded, before the slow release | 13 of 20 (13 first try) | `run_sort.py --episodes 20 --seed0 200 --ladder oracle` at commit 241a695 plus the return-home switch | `output/sort/win_home20_with` | Model accuracy. Fixed bins. |
| Same, return-home off, before the slow release | 17 of 20 (13 first try) | same with `--no-return-home` | `output/sort/win_home20_without` | The 13 vs 17 gap is within noise for 20 episodes. |
| First-try misses by bin and spot, before the slow release | rework bin third spot 6 of 6 missed; good bin 0 of 20; rework first two spots 1 of 34 | computed from the decision log | `output/sort/win_home20_with` | Only episodes that reached a third rework part exercise that spot (6). |
| Fill far side first (rejected) | 11 of 20 | spot order reversed, `--no-return-home` | `output/sort/win_far_first` | Not kept in the code. |
| Aim shifted 5 cm (rejected) | 10 of 20 | aim corrected by a measured carry, `--no-return-home` | `output/sort/win_carry_fix` | Not kept in the code. |
| Part moves 8 to 9 cm on release from the third spot; wrist within 1 cm of target; held part within 0.5 cm of the aim | 4 episodes, 22 placements | `output/bench/win_place_diag.py 204 205 206 208` | printed to the console, not saved | Four seeds chosen because they had failed. |
| Slow release fixes those four | 4 of 4 (was 1 of 4) | same script with `RAMP_S=1.0` (`output/bench/win_ramp.ps1`) | printed to the console, not saved | Same four seeds. |
| Clean sorts with the slow release, return-home on | 20 of 20 (17 first try), 52.3 s mean sim time for first-try episodes | `run_sort.py --episodes 20 --seed0 200 --ladder oracle` at d3ce846 | `output/sort/win_release_home` | Fixed bins. One set of 20 seeds. |
| Same, return-home off | 19 of 20 (18 first try), 49.4 s | same with `--no-return-home` | `output/sort/win_release_nohome` | Whether the models answer the in-bin question as well with the hand still near the bins. |
| Bins located from the camera | 40 of 40 found; error mean 0.5 mm, max 1.0 mm; bins moved 14.5 mm mean, 25.3 mm max | `run_sort.py --episodes 20 --seed0 200 --ladder oracle` at 9a991ec (bin jitter 0.02) | `output/sort/win_bins_jitter` (`episodes.jsonl`: `bins_true`, `bins_seen`) | Real depth cameras: the simulator's depth is exact, with no noise. |
| Clean sorts with moving bins the robot has to find | 19 of 20 (15 first try) | same run | `output/sort/win_bins_jitter` | The failure (seed 207) was a part dropped off the table mid-carry, not a detection error. |

## Setting parts down instead of dropping them (Oct 13, `output/bench/win_lower_diag.py`, good bin, one part per spot)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Part bottom above the bin floor at the hover (the drop) | 10.9 to 11.1 cm | `win_lower_diag.py 204` (probe mode) | `output/bench/win_lower.log` | |
| Lowest the wrist can go before the hand touches a wall top | 3.4 cm above the wall top (spot 0, near wall), 3.5 cm (spot 1, near wall), 2.7 cm (spot 2, far wall) | same | same | One seed. The hand (19 cm wrist to fingertip) straddles a wall of the 16 cm bin at every spot; it cannot enter the bin. |
| The part slides down in the pinch while the wrist descends 4 cm | 0.3, 0.6 and 1.0 cm (spots 0, 1, 2) | same, `part_bottom_above_floor` against `wrist_z` | same | The slide during the carry itself; not measured. |
| Drop with `lower` at 4 cm above the wall | 7.3 to 7.8 cm | `LOWER_MODE=0.04 win_lower_diag.py 204 205 206 207 208 209` | `output/bench/win_lower_040.log` | |
| Parts lying flat (tilt 90) at rest: hover vs lowered | 9 of 18 vs 6 of 18 | `LOWER_MODE=hover` and `LOWER_MODE=0.04`, same six seeds | `win_lower_hover.log`, `win_lower_040.log` | Six seeds, 18 placements each, parts released one after another into one bin. |
| Standing (tilt 10 or less) on the bin floor | 2 of 18 vs 2 of 18 | same | same | |
| Standing on top of another part | 1 of 18 vs 6 of 18 | same (`rest_z` 0.90 or more) | same | Counted as "in the bin" by the stacked-part allowance; a third part on two others is not. |
| Leaning (tilt 11 to 89) | 6 of 18 vs 4 of 18 | same | same | |
| Spot 0 part thrown 8.6 cm or more away from the robot (+x) and lying | 3 of 6 vs 4 of 6 | same, `rest_minus_aim_xy` | same | Spots 1 and 2 moved under 4.5 cm in every placement at both heights. Cause not investigated. The thrown spot-0 part is what the later parts land on. |
| In the bin at rest (geometry check) | 18 of 18 vs 18 of 18 | same | same | |

## The Oct 13 decision log, `test03` (the benchmark's test set; replaces test02)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Episodes and decisions | 90 episodes, 974 decisions (safe_to_grasp 300, in_bin 280, finish 270, sort_complete 124) | `run_sort.py --episodes 90 --seed0 400 --fault cycle --ladder oracle --out output/sort/test03` at d8b2d96 plus the resume change | `output/sort/test03` | Model accuracy: the oracle answered; this log is what the arms are replayed on. |
| Episode outcomes by fault | clean 30 of 30; forced drop 26 of 30; crowded 0 of 30 (by design: the held part is rejected, no dance) | same | `episodes.jsonl` | The 4 forced-drop failures (seeds 404, 422, 464, 473) are parts that left the table on the forced drop; nothing the line can re-feed. |
| Truth balance | finish good 101 / defective 169; safe_to_grasp grasp 240 / abort 60; in_bin yes 232 / no 48; sort_complete yes 56 / no 68 | computed from `decisions.jsonl` | same | |
| Wall time per episode | 28 s mean (headless, Windows, oracle) | same | same | test02 (Oct 11, under WSL) was 55 s per episode; the difference is not investigated. |

## Native Windows Ollama (Oct 13, 0.40.4-rc1 portable zip in `D:\Ollama`, models on `D:\Models\ollama`, port 11435)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Reading D: from WSL through the 9p mount | 251 MB/s (3.09 GB file, warm) | `cat` of a copied whisper blob, timed | printed to the console, not saved | Cold reads; writes. The reason Ollama went native: a 7 GB model would take about 30 s to load that way. |
| Gemma e4b load from D:, native Ollama | 6.6 s (load_duration) after an unload; 0.1 s wall for the warm one-word call | `D:\Ollama\smoke.ps1` | printed to the console, not saved | The very first load of the freshly copied file took 56.6 s (83.6 s wall), cause not investigated (the 6.6 s is the second cold load). One model, one try each. |
| Model placement | 43 of 43 layers on the GPU; 3.0 GB CUDA buffer plus 2.7 GB host buffer (Gemma's per-layer embeddings) | same | `D:\Ollama\serve.err.log` | `/api/ps` reports `size_vram` 0.5 GB for the same load; which figure is wrong is not investigated. |

## pplx-decider v1.1 27B, TurinTech Artemis GGUF, through the native Ollama (Oct 13, `D:\Models\pplx-decider\smoke.ps1`, `num_ctx 16384`, `num_batch 8192`)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Files verified | both SHA256 match the card (15.3 GB model, 0.93 GB BF16 projector) | `Get-FileHash` | printed to the console, not saved | |
| Ollama registers it as | capabilities `decision`, `vision`; architecture qwen35, 64 layers | `ollama create pplx-decider-artemis -f Modelfile`; `/api/show` | same | Whether Ollama runs the 16 full-attention layers noncausal as the original requires (the TurinTech card says its own llama-server check matched the original on 96.8% of 2,000 decisions; Ollama is not checked by anyone). |
| Text decision, no image, first call | 29.8 s wall including the load; good 0.716 / defective 0.284 | `smoke.ps1`, the test03 `finish` question and state text | same | One call. The state text does not describe the part, so there is nothing to decide from. |
| Image decision on `ep0400_000_finish.png` (truth good) | good 0.994; 3.8 s wall first, 0.8 s wall repeated | same | same | One frame. Latency through PowerShell's HTTP client, not the server's own figure (`latency_ms` came back empty). |
| Memory while loaded | GPU 22.7 GB of 24 used (model 13.1 GB + KV 0.96 + compute 2.6 GB, plus the desktop's 2.3); Windows commit 52 → 82 of 96 GB, the runner alone 30 GB; RAM free 33 → 29 GB | same, `Get-Process llama-server`, `nvidia-smi` | same | A long run: Gemma e4b's runner grew 9 → 18 GB of commit over 455 image decisions, so a 974-decision replay at this setting could exhaust the 96 GB commit limit. Not run. |

## pplx-decider 27B at the benchmark setting (`pplx-decider-artemis-ctx4k`, `num_ctx 4096`, `num_batch 4096`), memory, Oct 14 late (`D:\Models\pplx-decider\smoke.ps1 -Model pplx-decider-artemis-ctx4k`, native Ollama 0.40.4-rc1 on 11435, after the Oct 14 reboot: pagefile 64 GB, commit limit 128 GB)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| GPU memory for the loaded model | 19.1 GB (nvidia-smi 2,157 MiB before, 21,255 MiB loaded, 21,289 after two image calls; `/api/ps` size_vram 18.8 GB, Q4_K_M, ctx 4096) | `smoke.ps1`, one text and two image decisions | `output/bench/win_ctx4k_mem.log` | A long replay; the desktop's 2.1 GB is excluded. Larger than the 16.7 GB (13.1 + 0.96 + 2.6) the Oct 13 serve log listed at 16k context, so that breakdown undercounts; nvidia-smi is the figure to quote. |
| Windows commit while loaded | 46 → 70 GB (runner 23.9 GB); RAM free 33.0 → 31.2 GB; back to 46 GB after `keep_alive 0` | same, `Get-Process llama-server` | same | |
| Time | text 12.2 s wall including the load; image 0.6 s then 0.4 s wall (good 0.994 on `ep0400_000_finish.png`, truth good) | same | same | PowerShell client timing, `latency_ms` empty, as on Oct 13. |
| Both rungs of the chosen ladder on one 3090 | 4.7 GB (Decider 2B unit, `src/handoff/README.md`) + 19.1 GB + the desktop's 2.1 GB = 25.9 GB > 24 GB: they do NOT fit together on this card; the benchmark ran arms one at a time | arithmetic from the two rows | | Whether the 2B's 4.7 GB is still current (read from the unit on an earlier date); the live-ladder check will need the 2B on the WSL side and the 27B on the native Ollama, which share the card. |

## Viewer and build speed (Oct 11, demo scene, realtime pacing off, no panel, models idle)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Viewer, full robot meshes, 60 Hz, Windows | 97 steps/s (0.97x real time) | `output/bench/win_viewer.py output/bench/win_full60.yaml` | `output/bench/win_measure.log` | With the viewer on, the median step is 10.0 ms, the sim's timestep: about 100 steps/s is the viewer pacing to real time, not the machine's limit. |
| Viewer, full meshes, 30 Hz, Windows | 99 steps/s | `win_full30.yaml` | same | same |
| Viewer, quarter meshes, 30 Hz / 60 Hz, Windows | 100 / 91 steps/s | `win_q30.yaml` / `win_q60.yaml` | same | same |
| Headless, Windows | 271 steps/s | `win_headless.yaml` | same | |
| Same scene under WSL2 | 12 (viewer, full meshes), 74 to 85 (quarter meshes), 256 (headless) | the `measure*.py` scripts | `output/bench/measure*.log` | Measured Oct 11 under WSLg before the move. |
| Build time, Windows | 257 s cold, 35 to 41 s cached | the same runs | `output/bench/win_measure.log` | WSL was 24 to 27 s cached. |

## Checks after the move to Windows (Oct 11)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Grasp trials | fingers 10 of 10, weld 10 of 10, seeds 0 to 9 | `run_trials.py --trials 10 --grasp both` | `output/trials/win_s0` | Ten seeds. |
| Benchmark rescored on Windows equals the WSL result | identical except a path separator | `benchmark.py score --log output/sort/test02 --thresholds-from output/sort/log02b` | `output/sort/test02/benchmark.json` | |
| Models answer from the Windows side | Decider 2B 0.25 to 0.31 s warm; Gemma e4b 0.09 s warm | `output/bench/win_models.py` | printed to the console, not saved | Two questions, three calls each. |
| Head aims at the station and the bins | yaw -0.26 / -0.64 / -0.91 rad, pitch 0.68 to 0.76 | `output/bench/win_head.py` | `output/bench/win_head_*.png` | Checked by eye on three frames. |

## Model benchmark (Oct 11, under WSL, fixed bins)

The model and ladder numbers (test set `test02`, 131 decisions) are in `src/handoff/README.md` and `output/sort/test02/benchmark.json`. They were measured before the bins moved and before the slow release; re-run before quoting them as current.

## The finish question, by true finish (computed Oct 12 from the Oct 11 log)

36 finish decisions in `output/sort/test02` (14 good, 10 scratched, 12 unpainted). Each cell is what the model's top answer was. Computed from `decisions.jsonl` and `arms.jsonl` there; no thresholds applied.

| Model | Good parts | Scratched parts | Unpainted parts |
|---|---|---|---|
| Decider 2B with the frame | 12 good, 2 scratched | 10 scratched | 8 good, 1 scratched, 3 unpainted |
| Gemma e4b | 10 good, 4 scratched | 9 scratched, 1 good | 12 scratched |
| Gemma e2b | 14 scratched | 10 scratched | 12 scratched |

As a routing decision (good bin or rework bin): Gemma e4b sends 31 of 36 the right way and passes 1 defect as good; the Decider sends 26 of 36 the right way and passes 8 defects as good, all of them unpainted parts.

Does not show: anything on the current robot (fixed bins, old release, frames rendered under WSL); 12 unpainted parts is a small sample; whether a person finds the unpainted parts obvious (not tested).

## The finish question as good or defective (Oct 12, folded from the Oct 11 log)

The models were asked the three-way question on Oct 11. Here their two defect probabilities are added together (`collapse_finish`), which is what they would have sent to the rework bin. Command: `benchmark.py score --log output/sort/test02 --thresholds-from output/sort/log02b`. Result: `output/sort/test02/benchmark.json`; the three-way result is kept beside it as `benchmark.three_way.json`. 131 decisions, 36 of them finish.

| | Three-way | Two-way |
|---|---|---|
| Gemma e4b alone: right / confidently wrong | 0.733 / 0.145 | 0.824 / 0.145 |
| Decider 2B with the frame alone | 0.679 / 0.000 | 0.687 / 0.000 |
| Gemma e2b alone | 0.412 / 0.519 | 0.504 / 0.427 |
| Decider then person: right / confidently wrong / to a person | 1.000 / 0.000 / 0.786 | 1.000 / 0.000 / 0.786 |
| Gemma e4b then person | 0.985 / 0.015 / 0.687 | 0.970 / 0.015 / 0.481 |
| Decider then Gemma then person | 0.954 / 0.046 / 0.420 | 0.939 / 0.061 / 0.305 |

Does not show: how the models answer when actually asked the two-way question; anything on the current robot. Thresholds were chosen on log02b folded the same way.

## Gaps

- Three rows above were printed to the console and not saved to a file.
- Every multi-episode row is one run of 20 seeds.
- Nothing here measures the models with moving bins or with the slow release.

## All ten local arms on `test03`, scored Oct 14 (`benchmark.py score --log output/sort/test03`, thresholds and the split from this log's own held-out episodes)

Every arm answered all 974 decisions (`arms.jsonl`, 9,740 rows, 1 error in gemma-e4b). The table scores the 676 test decisions after a 30 percent by-episode held-out split (seed 0); intervals are 95 percent by-episode bootstrap (1,000 resamples). "Judgement" leaves out the 209 safe_to_grasp decisions in the test split, which a hard rule decides. Latencies are per call on the machine named and are NOT comparable across machines. Result file: `output/sort/test03/benchmark.json` (overwritten by every `score`). The decider-e4b-text arm ran from the WSL unit, the gemma `-self`, gemma-12b, decider-e2b-text and pplx-decider rows from native Windows on Oct 14 in chunks of 25 to 50 with the model unloaded between chunks (scratchpad `replay-chunked.ps1`).

| arm | n scored | right [95%] | confidently wrong [95%] | ECE | mean conf. | ms | judgement n / right / conf. wrong | ran on |
|---|---|---|---|---|---|---|---|---|
| decider-2b-text | 676 | 0.547 [0.52, 0.58] | 0.003 [0.00, 0.01] | 0.262 | 0.81 | 115 | 467 / 0.430 / 0.004 | WSL Decider unit (label added after the run) |
| decider-2b-vision | 676 | 0.780 [0.73, 0.83] | 0.000 [0.00, 0.00] | 0.078 | 0.72 | 325 | 467 / 0.767 / 0.000 | WSL Decider unit (label added after the run) |
| decider-e2b-text | 676 | 0.543 [0.51, 0.57] | 0.028 [0.02, 0.04] | 0.324 | 0.71 | 150 | 467 / 0.424 / 0.041 | native Windows (Ollama 0.40.4-rc1 on 11435, or the WSL text-Decider unit called from Windows) |
| decider-e4b-text | 676 | 0.562 [0.53, 0.60] | 0.034 [0.02, 0.05] | 0.253 | 0.80 | 184 | 467 / 0.452 / 0.049 | native Windows (Ollama 0.40.4-rc1 on 11435, or the WSL text-Decider unit called from Windows) |
| gemma-12b | 676 | 0.359 [0.31, 0.41] | 0.407 [0.36, 0.45] | 0.533 | 0.89 | 388 | 467 / 0.415 / 0.567 | native Windows (Ollama 0.40.4-rc1 on 11435, or the WSL text-Decider unit called from Windows) |
| gemma-e2b | 676 | 0.464 [0.44, 0.49] | 0.522 [0.49, 0.55] | 0.528 | 0.99 | 205 | 467 / 0.310 / 0.670 | WSL Docker Ollama 0.40.2 (label added after the run) |
| gemma-e2b-self | 676 | 0.470 [0.45, 0.49] | 0.530 [0.51, 0.55] | 0.478 | 0.95 | 357 | 467 / 0.319 / 0.681 | native Windows (Ollama 0.40.4-rc1 on 11435, or the WSL text-Decider unit called from Windows) |
| gemma-e4b | 676 | 0.768 [0.75, 0.79] | 0.195 [0.17, 0.22] | 0.191 | 0.95 | 162 | 467 / 0.749 / 0.197 | WSL Docker Ollama 0.40.2 (label added after the run) |
| gemma-e4b-self | 676 | 0.720 [0.70, 0.75] | 0.280 [0.25, 0.30] | 0.264 | 0.98 | 374 | 467 / 0.681 / 0.319 | WSL docker open-webui Ollama 0.40.2 (backfilled Oct 14, not recorded at run time); corner-pc Windows |
| pplx-decider-artemis-vision | 676 | 0.805 [0.76, 0.85] | 0.016 [0.00, 0.03] | 0.068 | 0.87 | 572 | 467 / 0.803 / 0.024 | native Windows (Ollama 0.40.4-rc1 on 11435, or the WSL text-Decider unit called from Windows) |

Does not show: the ladders (the `score` ladder list does not yet include the Artemis arm; the three ladder families are the next step), any cloud arm, and nothing about the live robot. gemma-12b's row is a prompt-following failure (it describes the frame correctly and still answers the cautious option, HISTORY.md Oct 14), kept as measured. The pplx-decider row is the `pplx-decider-artemis-ctx4k` tag (num_ctx 4096, num_batch 4096); its latency includes the image.

## Ladders on `test03`, three families, scored Oct 14 (`benchmark.py score --log output/sort/test03`, thresholds per rung and per question from the held-out episodes, applied to the 676 test decisions)

A = decision models reading the frame, no vision-language model anywhere. B = decision models reading the state text plus Gemma e4b's description of the frame (a VLM step by construction). C = Gemma itself as a rung. "Decided alone and right" is the share of all 676 the models answered without the person and got right; the person is the simulator's truth. Intervals are 95 percent by-episode bootstrap. The ladder's ms column in `benchmark.json` includes a nominal 5 s per hand-off and is not a latency measurement.

| ladder | decided alone and right [95%] | handed to a person [95%] | confidently wrong [95%] | right incl. the person |
|---|---|---|---|---|
| A: Decider 2B frame, then a person | 0.571 [0.54, 0.60] | 0.36 [0.34, 0.38] | 0.000 [0.00, 0.00] | 0.929 |
| A: Artemis 27B frame, then a person | 0.645 [0.62, 0.67] | 0.32 [0.29, 0.34] | 0.007 [0.00, 0.02] | 0.962 |
| A: Decider 2B, then Artemis 27B, then a person | 0.793 [0.75, 0.83] | 0.13 [0.11, 0.14] | 0.006 [0.00, 0.01] | 0.920 |
| B: Decider 2B text, then E2B text, then a person | 0.163 [0.14, 0.19] | 0.82 [0.79, 0.84] | 0.003 [0.00, 0.01] | 0.981 |
| B: Decider 2B text, then E4B text, then a person | 0.178 [0.15, 0.20] | 0.79 [0.76, 0.82] | 0.004 [0.00, 0.01] | 0.967 |
| B: Decider 2B frame, then E4B text, then a person | 0.620 [0.59, 0.65] | 0.28 [0.26, 0.31] | 0.001 [0.00, 0.00] | 0.904 |
| C: Gemma e4b, then a person | 0.521 [0.49, 0.55] | 0.41 [0.40, 0.41] | 0.068 [0.04, 0.09] | 0.926 |
| C: Decider 2B frame, then Gemma e4b, then a person | 0.786 [0.75, 0.82] | 0.13 [0.11, 0.15] | 0.009 [0.00, 0.02] | 0.919 |
| C: Artemis 27B, then Gemma e4b, then a person | 0.672 [0.64, 0.70] | 0.28 [0.26, 0.31] | 0.010 [0.00, 0.02] | 0.956 |
| C: Gemma e2b self, then Gemma e4b self, then a person | 0.675 [0.65, 0.70] | 0.00 [0.00, 0.00] | 0.325 [0.30, 0.35] | 0.675 |

What it shows: the two-rung frame-only ladder (A, 2B then 27B) and the Decider-then-Gemma ladder (C) are tied on decided-alone-and-right and on hand-off; A is confidently wrong on 0.6 percent against C's 0.9 percent, overlapping intervals. Does not show: anything live (every decision is on the oracle's path; the live-ladder check is still to run), any cloud rung, per-question rows (in `benchmark.json`), or what the thresholds do on a different log.

## Gemini 4 Argon (cloud, Kaggle Benchmarks) on a test03 sample, scored Oct 15

What ran: `gemini-4-argon` through Kaggle Benchmarks' model proxy (free quota, $1.72), on a private Kaggle dataset of 196 test03 decisions (18 whole episodes, seed 0) plus all 264 log02b decisions, each with its camera frame and the exact prompt the local self-confidence arms get (`scripts/kaggle_dataset.py`, `scripts/kaggle_task.py`, imported with `scripts/kaggle_import.py`). No log-probabilities: Argon states its own confidence in JSON. No per-call latency (Kaggle reports none).

Command: `benchmark.py score --log output/sort/test03 --subset-of argon-vision` (thresholds from test03's own held-out episodes, as for the Oct 14 headline). Every arm and ladder below is scored on the same 132 test decisions (12 episodes) that Argon answered. Result: `output/sort/test03/benchmark-subset-argon-vision.json`.

| arm (alone) | right | confidently wrong | ECE | mean confidence |
|---|---|---|---|---|
| Argon | 0.659 [0.59, 0.71] | 0.242 [0.18, 0.31] | 0.287 | 0.95 |
| Decider 2B, frame | 0.742 [0.61, 0.87] | 0.000 [0.00, 0.00] | 0.055 | 0.72 |
| pplx-decider 27B, frame | 0.750 [0.65, 0.85] | 0.023 [0.00, 0.07] | 0.112 | 0.86 |
| Gemma e4b | 0.735 [0.68, 0.79] | 0.220 [0.15, 0.29] | 0.222 | 0.95 |

| ladder (thresholds per rung and question) | decided alone and right | handed to a person | confidently wrong |
|---|---|---|---|
| Decider 2B, then Argon, then a person | 0.795 [0.70, 0.89] | 0.11 [0.08, 0.15] | 0.015 [0.00, 0.05] |
| Decider 2B, then 27B, then a person (local) | 0.773 [0.68, 0.87] | 0.14 [0.10, 0.18] | 0.015 [0.00, 0.05] |
| Argon, then a person | 0.371 [0.28, 0.45] | 0.61 [0.55, 0.68] | 0.015 [0.00, 0.05] |

What it shows: alone, Argon is confidently wrong on about a quarter of decisions (it says 0.95 to almost everything), like Gemma e4b and unlike the Decider. As the second rung after the Decider it ties the local 27B: the intervals overlap almost entirely, so the cloud rung buys nothing measurable over the on-device one on this sample.

What it does NOT show: n is 132 (12 episodes), so intervals are wide. Argon's thresholds come from only the 64 held-out decisions (6 episodes) it answered, and it is never trusted on safe_to_grasp or sort_complete (threshold 1.0). The same sample scored with log02b thresholds is in `benchmark-subset-argon-vision-thresholds-log02b.json`; those thresholds fit test03 poorly for every arm (Decider 2B hands 89% to the person there), so they are not the comparison. Argon has no Robotics-ER-style tuning here; it got the same text prompt as every arm.

## Joint threshold fit (arXiv 2610.21412's per-choice rule) against the greedy fit, test03, Oct 16

What ran: `scripts/compare_threshold_fits.py` (offline, no model calls). The greedy fit (the headline's) sets each rung's threshold as if what it passes up goes to the person; the joint fit searches every rung's threshold together on the same objective, so a lower rung passes up what the next rung does better (`choose_thresholds_joint`, after docs/papers/README.md). Same 974 test03 decisions, held-out fraction 0.3, five splits (seeds 0 to 4), mean of the test-side rates.

| ladder | fit | decided alone and right | handed to a person | confidently wrong |
|---|---|---|---|---|
| Decider 2B, then 27B | greedy per rung | 0.768 | 0.109 | 0.007 |
| Decider 2B, then 27B | greedy per question | 0.716 | 0.207 | 0.006 |
| Decider 2B, then 27B | joint per rung | 0.653 | 0.268 | 0.019 |
| Decider 2B, then 27B | joint per question | 0.668 | 0.289 | 0.009 |
| Decider 2B, then Gemma e4b | greedy per rung | 0.657 | 0.207 | 0.022 |
| Decider 2B, then Gemma e4b | greedy per question | 0.678 | 0.249 | 0.006 |
| Decider 2B, then Gemma e4b | joint per rung | 0.657 | 0.211 | 0.017 |
| Decider 2B, then Gemma e4b | joint per question | 0.673 | 0.267 | 0.033 |

What it shows: the joint fit does not help. On the headline ladder it is worse on all three measures; on the Gemma ladder it is a wash. It wins on the decisions it is fitted on by construction (a test asserts that), so the loss on test is overfitting: about 290 held-out decisions cannot support searching both thresholds together. The greedy fit stays the default.

Side finding that affects the headline: the headline ladder (greedy, per question) scores 0.793 decided alone and right with 0.13 to a person on seed 0, the split every published figure uses. Over five splits it averages 0.716 and 0.207, so seed 0 is a favourable split. Confidently wrong holds at 0.6% either way. The headline should carry a several-split figure or say it is one split.

What it does NOT show: no intervals (five splits' means, not bootstraps); only two ladders (the 27B ladder has no log02b replay, so no cross-log row); the paper's call-cost term is left out (no rung costs anything here but time). `benchmark.py score` now also prints the joint modes; the earlier modes' numbers are unchanged (checked against the pre-change `benchmark.json`).

## Headline ladder over ten splits and five person costs, test03, Oct 16

What ran: `benchmark.py robust --log output/sort/test03 --splits 10 --human-costs 0.2,0.35,0.5,0.65,0.8` at ef46a89 (offline). Ladder: Decider 2B frame, then pplx-decider 27B frame, then a person; thresholds per rung and question, fitted on each split's held-out episodes (30%), scored on the rest. Mean over the ten splits. Result: `output/sort/test03/benchmark-robust.json` (also the per-rung and thin-question-fallback fits, every ladder, and each split's numbers).

| person cost | decided alone and right | to a person | acted wrong, all | acted wrong, judgement | confidently wrong, judgement |
|---|---|---|---|---|---|
| 0.2 | 0.566 | 0.405 | 0.030 | 0.035 | 0.011 |
| 0.35 | 0.634 | 0.322 | 0.043 | 0.037 | 0.011 |
| 0.5 (the headline's) | 0.742 | 0.170 | 0.088 | 0.054 | 0.011 |
| 0.65 | 0.766 | 0.119 | 0.115 | 0.079 | 0.005 |
| 0.8 | 0.763 | 0.100 | 0.137 | 0.111 | 0.006 |

"Acted wrong" = answered without the person, and wrong, at any confidence. "Judgement" leaves out safe_to_grasp, which a hard rule decides; on it the models answer everything at the base rate (0.81 right on seed 0), so it inflates both "decided alone" and "acted wrong".

What it shows: the published headline (seed 0: 0.79 alone, 0.13 to a person, 0.006 confidently wrong) leaves out that the ladder acts wrongly on 8% of all decisions on that split (1 - 0.793 - 0.127), 3% of the judgement ones; most of those wrong answers sit below 0.9 confidence, so "confidently wrong" does not count them. The person cost moves every number: the trade-off is a curve, and the point on it should come from what a wrong action costs on a real line, not from 0.5.

What it does NOT show: ranges across splits are spreads, not confidence intervals (splits share decisions); nothing live; the cost of a wrong action is held at 2 (right +1, wrong -1) while only the person's cost moves.

## The confirmation log, `test04` (Oct 16 to 17; the frozen ladder's one arm replayed Oct 19, see the next section)

| Figure | Value | Command | Raw log | Does not show |
|---|---|---|---|---|
| Episodes and decisions | 300 episodes (seeds 1000 to 1299, all new), 3,262 decisions (safe_to_grasp 1,000, in_bin 949, finish 900, sort_complete 413), every one with a truth, no duplicates | `run_sort.py --episodes 300 --seed0 1000 --fault cycle --ladder oracle --out output/sort/test04` at 942a7e6 to c73ab07 (stopped at 283 when a session closed; the same command resumed it) | `output/sort/test04` | Model accuracy: the oracle answered. Frozen per docs/PREREG.md: no arm is replayed on it until the ladder and thresholds are written there. |
| Episode outcomes by fault | clean 96 of 100; forced drop 93 of 100; crowded 0 of 100 (by design) | same | `episodes.jsonl` | The 4 clean failures are new (test03 was 30 of 30); not investigated yet. |

## Confirmation on `test04`: the frozen ladder, scored once, Oct 19 (`scripts/freeze_ladder.py confirm` per docs/PREREG.md at 1c6c475)

Ladder frozen in docs/PREREG.md (committed 1c6c475 before any test04 replay): `d3-27b-vision` > person, one threshold 0.80 fitted on all of test03, `human_cost` 0.2. Only d3 27B was replayed on test04: 3,262 of 3,262 decisions answered, no errors (same GGUF files, server and flags as the PREREG lists; Windows commit 101.5 GB at load). No refit. Each rate: test04 value with its 95% by-episode bootstrap interval (300 episodes), beside the test03 prediction (mean over the 10 held-out splits' test episodes).

| | judgement questions (2,262) | test03 prediction | all questions (3,262) | test03 prediction |
|---|---|---|---|---|
| Acted wrong (machine answered alone, wrong) | **0.026** (0.017 to 0.035) | 0.018 | 0.032 (0.024 to 0.039) | 0.018 |
| Confidently wrong (acted wrong at 0.9 or more) | 0.019 (0.012 to 0.025) | 0.014 | 0.013 (0.008 to 0.018) | 0.010 |
| Handed to the person | 0.185 (0.168 to 0.202) | 0.207 | 0.233 (0.215 to 0.251) | 0.311 |
| Decided alone and right | 0.789 (0.770 to 0.809) | 0.775 | 0.735 (0.716 to 0.755) | 0.671 |

By question, test04 (test03 all 974 at the same threshold in brackets): finish person 0.109 / acted wrong 0.016 (0.126 / 0.019); in_bin 0.068 / 0.047, confidently wrong 0.038 (0.075 / 0.032, 0.032); safe_to_grasp 0.342 / 0.046 (0.363 / 0.030); sort_complete 0.620 / 0.000 (0.621 / 0.000). Raw: `output/sort/test04/confirmation.json`, `output/sort/test04/arms.jsonl`.

What it shows: on the judgement questions (the headline, per PREREG) the frozen ladder acted wrong on 2.6% of decisions (1.7 to 3.5) and decided 78.9% alone and right; the test03 predictions for acted wrong, confidently wrong and alone-and-right all fall inside the test04 intervals. It handed slightly fewer decisions to the person than predicted (18.5% against 20.7%, prediction just above the interval). Over all questions it did worse than predicted: acted wrong 3.2% against 1.8%, outside the interval (Oct 21: mostly a scoring artifact; live-equivalent it is 1.8%, see the next section), from safe_to_grasp (4.6% acted wrong against 3.0%; the question a hard rule decides before any model in the running system) and in_bin (4.7% against 3.2%; 3.8 of the 4.7 points, about 80%, at 0.9 confidence or more). The by-question gap on in_bin is the one to look at: d3 is mostly confidently wrong there, and no threshold below 0.9 can catch that part.

What it does NOT show: live behaviour (every decision sits on the oracle's path); other ladders on test04 (only the frozen arm was replayed, by design); why in_bin errors rose (not investigated); real cameras or parts. The test03 prediction is itself an average with spread (judgement value 0.899 to 0.947 across splits), and it is the winning candidate's own score, the best of 525 ladder × fit candidates on the same ten splits it was picked on, so it could be optimistic (winner's curse). **Measured, it is not** (`scripts/nested_pick.py`, Oct 20): over 10 outer splits of test03 (pick on 70% of episodes, freeze on that 70%, score once on the other 30%), predicted minus observed was judgement acted wrong +0.006 (sd 0.018), judgement value -0.012 (sd 0.038), handed to the person 0.000 (sd 0.036); the pick was `d3-27b-vision` alone every time, so there was little to select among. So the winner's curse does not explain test04's higher acted-wrong rate (2.6% against 1.8%). What does is not established; the outer splits alone ranged from 0.0% to 3.9% acted wrong on 27 episodes each. Raw: `output/sort/test03/nested-pick.json`.

## Live-equivalent rescoring of the confirmation: hard rules answer first (post-confirmation note, Oct 21)

Found by a review agent on Oct 21 and checked with `scripts/live_equivalent.py` (committed). The confirmation (`freeze_ladder.py confirm`) composed the frozen ladder over every logged decision, including the ones the sim's hard rules answered while the log was made (`answered_by == "rule"`: a part on the floor or a crowded station forces safe_to_grasp to abort, and a part on the floor forces sort_complete to "no"). In the running system `Ladder.decide` returns the rule's answer before d3 is asked, so d3 never sees those decisions.

Method: the same frozen thresholds and the same 95% by-episode bootstrap. Each rule-decided row is replaced by the rule's own logged answer (machine-answered, not a hand-off). No frozen value changes. The rule's answers agree with the truth on every one of them, in both logs.

```
.venv\Scripts\python.exe scripts\live_equivalent.py --log output\sort\test04 --frozen output\sort\test03\frozen-ladder.json
```

This writes `output/sort/test04/live-equivalent.json`; with `--log output\sort\test03` it writes the test03 one.

What moved on test04:
- 200 safe_to_grasp decisions: the replay had scored 46 as d3 acting wrong and 154 as hand-offs.
- 20 sort_complete decisions: the replay had scored all 20 as hand-offs.

On test03: 60 safe_to_grasp (9 scored as acted wrong, 51 as hand-offs) and 8 sort_complete (all hand-offs).

| test04 | as replayed (the confirmation) | live-equivalent |
|---|---|---|
| all questions, acted wrong | 0.032 (0.024 to 0.039) | **0.018** (0.012 to 0.024) |
| all questions, handed to the person | 0.233 (0.215 to 0.251) | **0.180** (0.167 to 0.192) |
| all questions, alone and right | 0.735 | 0.802 |
| judgement, acted wrong (the headline) | 0.026 (0.017 to 0.035) | 0.026 (unchanged) |
| judgement, handed to the person | 0.185 (0.168 to 0.202) | 0.176 (0.159 to 0.193) |

Like for like on test03 (whole log, frozen thresholds, so in-sample): all questions acted wrong 0.024 as replayed, 0.014 live-equivalent; judgement 0.021 both ways.

What it shows:
- **The all-questions gap was mostly a scoring artifact.** The confirmation's "3.2% against a 1.8% prediction, outside the interval" counted 46 safe_to_grasp errors the running system cannot make.
- **Live-equivalent, test04 acts wrong on 1.8% of all decisions**, in line with test03's live-equivalent 1.4% (inside the interval).
- **About a fifth of the all-questions hand-offs** were rule cases that never reach the operator page.
- **The judgement headline (2.6%, by PREREG) does not change.** The judgement hand-off rate moves by about 1 point, from the 20 sort_complete rows.

What it does NOT show:
- The test03 ten-split prediction rescored the same way: the prediction stays as frozen. This note compares like with like only through the whole-log test03 figures.
- Anything about why judgement acted wrong rose from 1.8% to 2.6%. That is still not established.

## Where the frozen ladder hands off and acts wrong on `test04`, by fault type (post-hoc re-cut, Oct 19)

Not a new claim: the same frozen ladder (`d3-27b-vision` > person, threshold 0.80) and the same test04 answers as the confirmation above, split after the fact. Reproduce with `scripts/handoff_breakdown.py --log output/sort/test04 --frozen output/sort/test03/frozen-ladder.json` (writes `output/sort/test04/handoff-breakdown.json`; the wrong-bin table below is its `in_bin_truth_no_by_where`, and `--log output/sort/test03` gives the test03 column). Definitions: an episode's fault type is `fault` in `episodes.jsonl` (none = clean; 100 episodes each, 3 parts each); "handed to a person" = d3's top probability under 0.80; "acted wrong" = at or over 0.80 and not the truth; judgement = every question but safe_to_grasp.

| fault type | judgement handed to a person | judgement hand-offs per 3-part episode | episodes with no judgement hand-off |
|---|---|---|---|
| clean | 9.2% | 0.67 | 52% |
| drop_on_carry | 12.5% | 1.03 | 29% |
| crowded (cannot be finished, by design) | 35.3% | 2.49 | 0% |

By question on clean episodes: sort_complete handed off 33%, safe_to_grasp 23% (rule-decided in the running system), finish 8%, in_bin 2.5%; acted wrong 0 on every question except in_bin (8.9%: 29 of 327, 26 of them at 0.9 confidence or more, 3 between 0.80 and 0.90). Crowded episodes hand off sort_complete 99% of the time, which is the right call for a sort that cannot be finished.

**The in_bin misses are wrong-bin landings.** Every in_bin error on test04 is the machine saying "yes" when the truth is "no". Split by where the part actually was (`context.parts[part].in` in `decisions.jsonl`, the logged geometry check at that question):

| truth "no" because the part was... | test03 (the log the threshold was fitted on): d3 said yes / handed off / said no | test04: d3 said yes / handed off / said no |
|---|---|---|
| in the OTHER bin (wrong bin) | **8 / 0 / 0** | **29 / 0 / 0** |
| in no bin (table, floor, hand) | 1 / 20 / 19 | 16 / 65 / 70 |

d3 called every wrong-bin landing "in the bin", every one at 0.9 confidence or more, on both logs (test04's 29 are 29 in_bin decisions on 15 landings: after a "no" the tree asks again, so most landings were asked twice; counted Oct 27 from `context.parts[part].in` against `context.intended_bin`. Rounding fixed the same day: three printed values above had rounded 0.0115, 0.19245 and 0.0915 down; the raw files are unchanged) (example: test04 ep1009 decision 104, a scratched part meant for rework resting in the good tray, d3 "yes" at 0.99). It sees a part in a tray and does not tie it to the tray it was meant for. A defective part in the good bin is the costliest error this cell can make, and no confidence threshold catches it. Parts in no bin are mostly caught (135 of 151 handed off or answered "no").

What it does NOT show: whether a prompt that names the intended tray's colour more strongly would help (not tried; test04 is spent for claims); how often wrong-bin landings happen on a real line.

**Prompt audit of those wrong-bin answers (Nov 2; a reading of the code and logs, no model was run).** Question: was d3 ever asked the question this result blames it for failing?

Method: read `Question("in_bin")` in `src/handoff/decider.py:59` and the `state` and `criteria` it sends in `src/handoff/rungs.py:115`; read `state_text` on the logged wrong-bin decisions in `output/sort/test04/decisions.jsonl`; counted wrong-bin in_bin decisions per log from `context.parts[part].in` against `context.intended_bin`.

- d3 WAS told the target. The logged state on a wrong-bin decision says: "The hand released the part over the rework bin, the orange (left) tray, about 1.5 s ago ... the orange tray on the left is the rework bin and the green tray on the right is the good bin."
- The answer options do not allow for the case. `yes` = "the part is resting inside the bin walls". `no` = "the part is outside the bin, on the table, floor, or still in the hand". A part in the other tray fits `yes` and fits no wording of `no`.
- The state also asserts the release happened over the intended tray, which is false on these landings.

So "d3 said yes" is partly the answer the criteria ask for, and "cannot see that error" went further than this evidence. The retest below settled which part of this mattered: the criteria did not.

Frames available to test it: test03 8, log02 2, scale-wrongbin 2 (log02b also counts 2, not checked for overlap with log02). test04's 29 are spent for claims.

**Wrong-tray retest: one frame is the gap (Nov 2; exploratory, NOT a scored claim).** Question: does more context let d3 catch a part in the wrong tray? dup asked whether d3 had ever been given the right tray; that question led to this run.

Method: `.\run.ps1 python scripts/probe_wrong_tray.py run` re-asked d3 27B (local, threshold 0.80) on the saved in_bin frames of test03 and scale-wrongbin, 288 decisions and 1,796 answers, 0 errors. log02 left out: its state never names the tray colours. Answers in `output/sort/wrong_tray_probe/`. `baseline` matched the logged d3 answer on all 280 test03 decisions (largest difference 0.0; `score` prints this and refuses to write results above 0.01). The split by target tray: `probe_wrong_tray.py occupancy --logs test03 test04 scale-wrongbin`, no model run: test03 and test04 use their logged replays; scale-wrongbin has no d3 replay, so its row is this probe's baseline re-ask.

**The split:** every wrong-tray "yes" had an earlier part already in the target tray; both "no" answers had it empty.

| log | target tray already held a part | target tray empty |
|---|---|---|
| test03 | 8 decisions on 4 landings, all said yes | none |
| test04 | 29 decisions on 15 landings, all said yes | none |
| scale-wrongbin | none | 2 decisions on 2 landings, both said no |

**Prompts, decisions at 0.80:**

| prompt | wrong tray: yes / handed off / no | no tray: yes / handed off / no | correct tray: yes / handed off / no |
|---|---|---|---|
| `baseline`: production prompt | 8 / 0 / 2 | 1 / 20 / 19 | 237 / 1 / 0 |
| `criteria`: `no` also covers the other bin | 8 / 0 / 2 | 1 / 23 / 16 | 234 / 3 / 1 |
| `criteria_neutral`: and drop the claim it was released over the target | 4 / 4 / 2 | 1 / 1 / 38 | 182 / 45 / 11 |
| `counts`: and state each tray's count before the release | 2 / 2 / 6 | 0 / 4 / 36 | 211 / 24 / 3 |
| `before_after`: an earlier frame of the trays, plus the current one | **0 / 0 / 8** | 0 / 0 / 5 | **147 / 1 / 0** |

**Controls: same text as `before_after`, only the frames change (161 decisions).** Added because nothing showed d3 had read both images.

| frames sent | wrong tray: yes / handed off / no | no tray: yes / handed off / no | correct tray: yes / handed off / no |
|---|---|---|---|
| before, then now (`before_after`) | 0 / 0 / 8 | 0 / 0 / 5 | 147 / 1 / 0 |
| now, then before (`ba_swapped`) | 2 / 6 / 0 | 0 / 1 / 4 | 35 / 32 / 81 |
| now, twice (`ba_same_twice`) | 0 / 8 / 0 | 0 / 1 / 4 | 42 / 93 / 13 |
| now only (`ba_one_frame`) | 2 / 6 / 0 | 0 / 1 / 4 | 67 / 67 / 14 |

What it shows: the frames drive the result, not the prompt's wording. With the same text, correct-tray "yes" falls from 147 to 35 when the frames swap, so d3 reads both images and in order. On these frames the criteria were not the cause (8 yes before and after the change), and d3 can tell a wrong-tray landing once it can compare two frames. A single frame does not say which part just landed.

What it does NOT show:
- A rate. `before_after` has 8 wrong-tray decisions on 4 landings, because a landing needs an earlier part's frame in the same episode (scale-wrongbin's 2 have none; only 5 no-tray and 148 correct decisions qualify).
- Anything clean about test03: it chose the ladder, and these prompts were written after I had looked at its frames. test04 was not re-asked. A claim needs test05.
- Live behaviour. No run captures a before frame; the earlier frame here is the trays after the previous part's release, and `counts` used the logged truth, the best case for what the robot could know.
- The frozen ladder's results change in any way. They stand as scored, on the one-frame prompt.

## Exploratory: small-model-first cascades on `test03` (Oct 21; not a claim, test03 chose the ladder)

The question: would a 2B > 4B > 9B > 27B cascade beat d3 27B alone (dup's 2-4-9-27 idea)? The ladder pick only searched the 10 best arms alone, so no 2B or 4B model was ever a rung. Scored here the same way (`scripts/cascade_check.py`, CPU only: same 10 splits, same three threshold fits with the best one kept, human_cost 0.2, judgement questions).

| ladder | judgement value | handed to the person | acted wrong | machine ms |
|---|---|---|---|---|
| d3 27B alone (frozen) | **0.923** | 20.7% | **1.8%** | 277 |
| d3 2B > 4B > 9B > 27B | 0.884 | 10.9% | 4.7% | 346 |
| d3 2B > 27B | 0.896 | 14.5% | 3.8% | 268 |
| d3 9B > 27B | 0.895 | 9.9% | 4.3% | 325 |
| Intern 2B > 4B > d3 27B | 0.907 | 12.4% | 3.4% | 355 |
| Intern 2B > d3 27B | 0.919 | 12.5% | 2.8% | 333 |
| Decider 2B > d3 27B | 0.924 | 20.1% | 1.8% | 599 |

What it shows:
- **Cascades halve the hand-offs, but the small models act wrong more often.** At human_cost 0.2 (a wrong action costs 10 hand-offs, dup Oct 16) they lose.
- **The full d3 family cascade is the worst.** d3 2B alone answers 55% of decisions and is confidently wrong too often.
- **Intern 2B > d3 27B is the one near-tie:** 0.004 below, inside the pick rule's 0.005. It trades 8 points fewer hand-offs for 1 point more wrong actions. The fewest-rungs tie-break would still pick d3 alone.
- **Decider 2B > d3 27B scores marginally higher (+0.001),** but the 2B answers only 1.5% of decisions, so it is mostly a 320 ms toll on every one.
- **Speed is no argument here:** on the 3090 the 27B (277 ms) is as fast as the small ones, and a 2 + 4 + 9 + 27 stack does not fit in 24 GB beside the sim anyway.

What it does NOT show: anything on a fresh log (test03 picked the ladder; these are in-sample comparisons to it). Any change to the ladder would need a PREREG entry and test05.

## Measured: what each MH3 joint turns (Oct 21, for the dashboard's joint labels)

Asked: label the arm sliders on the MH3 page (dup, Oct 21; they read `left_1` to `left_7`). The model file is binary USD and its joints carry only numbers, so each joint's job was measured rather than guessed.

Method: `scripts/probe_joint_axes.py` (Genesis CPU, cameras off, output `output/probe/joint_axes/<ts>/axes.json`). For each public joint, from the zero pose and from the kit home pose, teleport it +0.1 rad with no physics step and read the rotation of the hand's root link (the head camera mount for head joints, the joint's own tip for fingers and thumbs). Axes are in the torso frame (`torso_base`: x to the robot's left, y backward, z up; read from where the arms sit: left arm at +x, the hands at -y), finger and thumb axes in the hand frame (`gripper_base`: fingers along +z, curling toward -x).

| Joint (both arms the same) | Axis at zero pose | Hand moved per 0.1 rad | Label |
|---|---|---|---|
| 1 | left-right | 58 mm | Shoulder pitch |
| 2 | front-back | 53 mm | Shoulder roll |
| 3 | vertical (arm hanging) | 24 mm | Shoulder yaw |
| 4 | left-right, at the elbow (link 4 sits 32 cm below the shoulder) | 32 mm | Elbow |
| 5 | along the forearm (parallel to elbow-to-wrist within a few degrees) | 0 mm, the hand turns in place | Forearm twist |
| 6 | across the forearm, vertical at home | 3.5 mm | Wrist yaw |
| 7 | across the forearm, left-right | 1.2 mm | Wrist pitch |
| thumb_1 | the hand's long axis; the tip swings toward the palm side | | Thumb swing |
| thumb_2 | about the palm normal; the tip moves toward the fingers | | Thumb close |

Head: `head_yaw` turns about the vertical, `head_pitch` about left-right, as named. What it does NOT show: the names "yaw" for joints 3 and 6 hold at the poses measured (arm hanging, and at home); with other joints moved, an axis points elsewhere, as on any serial arm.

## Measured: the carry, its crossings and the closure (Oct 21, Genesis CPU, seed 7's three parts; one process per run, `gs.cpu`, decimated robot, cameras off)

What was asked (dup, Oct 21, after the demo at 1:44 PM ET, seed 7, `--scales --fault wrong_bin --close 0.75`): the demo "is stuck in a loop dropping the last part", and "the arm movements have too many steps and long pauses between steps". The demo's log: part_12 fell out of the hand during the carry four times (the frames show it on the table past the good tray), each grip reading the thumb at 0 N; the run ended in failure with two bumps on part_10's carry; the sim ran at about 0.4 x real time (6 s of sim in 15 s of wall, from the bump stamps).

Method: `scripts/probe_carry.py --seed 7 --closes 0.75,0.85,0.9` runs drive_to_table and look_at_table, then reach, approach, grasp (fingers, fixed closure), lift, move_to_bin, lower, release and retract for each part of seed 7 (target tray by truth: all three are rework), recording each skill, where the part left the hand (contact solver: truth, to measure only), the fingertip forces after the grasp, every bump stop with what the contact solver saw at that moment, and per wrist move its planned (eased) time and the settle tail. `scripts/probe_crossing_trace.py --seed 7 --part 2 --close 0.85` runs the same carry with the guard's stop switched off and records every physics step of move_to_bin and retract: hand IMU statistic, wrist position and speed, each arm joint and its distance to its limits, and every contact of any robot link.

| Run (closure 0.85 unless said) | What it showed |
|---|---|
| carry, joint-space moves (as of 279a4d8) | 0.75: **0 of 3 carried**, every part left the hand in move_to_bin, thumb 0 N. 0.85: 3 of 3 (thumb 2.2 to 5.0 N). 0.9: 3 of 3 (thumb 8.8 to 11.5 N). Every carry and every retract stopped on a bump mid-crossing: 15 wrist moves and 14.5 s of sim per cycle, 6.5 s wall on CPU |
| trace, joint-space moves | **the crossing sagged**: planned at the clearance height (about 1.03 m), the wrist went down to 0.945 m and the gripper base dragged over the good tray's and the rework tray's wall tops (contact solver) at 18 to 26 m/s^2 on the hand IMU. Joint-space interpolation between two poses at one height bows the hand off that height; the stop-time truth check missed it because the contact was over by the stop |
| trace, straight-line moves (IK every 1 cm, grasp orientation held) | the wrist stayed at 0.99 to 1.07 m with no tray contact, but the hand IMU still stopped every crossing at 3.2 to 6 m/s^2: `left_2`, `left_6` and `left_7` reached their limits (the wrist joints turn 0.3 rad each way) and the IK switched pose mid-line, joints 5 and 6 counter-rotating at 2 to 4 rad/s while the hand barely moved |
| carry, straight-line crossings with `free_yaw` as first built (**see the correction below: this mask froze the yaw and freed the pitch**), lower with the full orientation | 0.85 and 0.9: **3 of 3 carried, 0 bumps**, 10 wrist moves, **12.2 s of sim per cycle, 4.4 s wall** (CPU). 0.75: 1 of 3 carried. The hand's heading stayed within 2 degrees of the grasp yaw through the carry |
| carry, `lower` with free yaw too | the hand touched the rework tray's near wall on the way down every time (truth contact), stopping the descent; with lower holding the full orientation it did not, so only the crossings use free yaw |

**Correction (same day, from the review workflow wf_8bfc1f67, checked against Genesis 1.4.3's source):** the first `free_yaw` passed the IK the hand's own up axis (`[False, True, False]`), but Genesis masks the rotation error in WORLD axes and inverts a one-axis mask, so it held the world-x and world-z error and freed the rotation about world y: the heading stayed held (hence "within 2 degrees") and the hand could pitch fore and aft. The rows above marked free yaw were measured with that mask. With the true yaw mask (`YAW_FREE_MASK = [False, False, True]`, 1d10856's follow-up) see the next section's last rows.

What it shows: the closure 0.75 is not a grasp (the thumb never reaches the part), so it is out; the two candidates are 0.85 and 0.9. The extra steps and stops were the bump guard catching real wall contacts made by a sagging crossing, then false stops made by the wrist hitting its joint limits; straight-line, level, yaw-free crossings remove both, and the cycle is 10 moves instead of 15.

What it does NOT show: anything on CUDA or in the demo's viewer (the next demo run does that, with a pace figure on every event line); seeds other than 7 or more than three parts; whether 0.85 holds through the force of a faster carry; the 0.4 x real-time pace of the GPU demo, which these CPU runs do not measure. The behaviour change that came with it: after one failed robot re-pick of a part, the operator places it (`MAX_ROBOT_REPICKS`, sort.py), tested on the fake simulator.

## Measured: the touch grip against the fixed closure, seeds 1 to 10 (Oct 21, Genesis CPU, 30 carries per row)

What was asked (dup, Oct 21): "can we not just make it not clip instead? ... it wouldn't matter with a metal part irl how hard it grips it." The fixed closure commands the fingers inside the part and Genesis's soft contact lets them sink in; a real metal part does not give. The touch grip (`--grip touch`: `grasp_control="force"`, `grasp_mode="weld"`, thumb closed on its own to 0.5 N) closes until the fingertip sensors feel the part, rests the fingers where they stopped, and welds the part to the hand: the weld stands in for a rigid part's friction grip. It is a simulation stand-in, not a model of the real grip.

Method: `scripts/probe_carry.py --seeds 1-10` (each seed's three parts squared up in the station, carried to the tray its truth says; drive_to_table and look_at_table first), rows below; "clean" is every skill ok; a bump stop is the hand IMU stopping a guarded move, "real" when the contact solver (truth, to measure only) shows the hand on a tray or the table at the stop; "sink" is the deepest hand-link penetration into the part after the grasp, lift, move_to_bin or lower.

| Row | Clean | Failures | Carries with a bump stop (real at the stop) | Deepest sink | Mean cycle (sim) |
|---|---|---|---|---|---|
| fixed 0.9, motion as of 279a4d8 (baseline worktree) | 20 | 4 dropped on the carry, 4 closed on nothing, 2 approach missed by 21 mm | 17 (34 stops, 13 real) | 15.8 mm | 12.0 s |
| fixed 0.9, motion of 1d0a0e8 | 19 | the same 4 + 4 + 2, 1 hover missed (gave up after bump retries) | 6 (10 stops, 0 real) | 15.8 mm | 10.5 s |
| touch, keeper on after the weld | 24 | 2 approach, 1 lift, 1 lower, 2 hover | 16 (33 stops, 9 real) | 3.9 mm | 14.0 s |
| touch as built (keeper off after the weld, fingers rested at their measured closure, weld-aware held check) | **26** | 2 approach missed by 21 mm, 2 hover missed (bump retries ran out) | 16 (33 stops, 7 real) | **4.0 mm** | 14.3 s |
| **with the true yaw mask** and an IK branch-flip fallback: touch | **27** | 2 approach missed by 21 mm, 1 part not resting in the tray after the release | **4 (4 stops, all 4 real)** | 4.0 mm | 14.0 s |
| **with the true yaw mask**: fixed 0.9 | 20 | (per-carry file lost: both runs started in the same second and shared an output folder; numbers from its summary line) | **1 (1 stop, not real)** | 16.5 mm | 10.5 s |

Along the way: the force grip's own check (finger tips still at 0.93 N after a short hold) refused 23 of 30 touch grips, which read 0.4 to 0.7 N; under a weld the touch counts once the fingertips felt the part and the hand is not closed on itself. The lift's "held" check needed finger contact, which fingers resting at the surface can lack: a welded part near the wrist and off the table now counts. Resting the fingers at their measured closure after the weld took one run's stops from 33 to 19; the rerun with the weld-aware check was back at 33, so that effect is not established.

All rows above the last two ran with the first `free_yaw` mask, which froze the yaw and freed the pitch (correction in the previous section); most of their bump stops went with it. With the true mask the hand's largest tilt from level while carrying was 10 degrees (both grips; per-step `max_tilt_deg` in the probe).

What it shows: the motion change of 1d0a0e8 did not cause the failures seen on seeds 1 to 10 (the baseline has the same drops, empty closes and approach misses) and cut the bump stops from 17 carries to 6. The touch grip sinks 4 mm at most against 16 and carried 26 of 30 against 19, with no drops and no empty closes. With the first mask it stopped on a bump in about half its carries and both grips had false stops in the first 10 to 18 steps of a move at 5 to 12 m/s^2; with the true yaw mask the touch grip stopped in 4 of 30 carries, each on a real wall touch, and fixed 0.9 in 1 of 30, a false one.

What it does NOT show: anything on CUDA or in the viewer (the 3:14 PM touch demo ran with the first mask: 3 of 3 sorted, 8 bump stops); why the touch grip touches the rework tray's near wall (a less closed hand hangs lower: not measured); why both grips see early-move IMU spikes; the right arm; the part tilting in the hand under the weld; how many of the true-mask carries hit the IK branch-flip fallback (the probe did not record it then; since the Oct 22 nightly review's fix it records each in `line_fallbacks`, and the fallback now eases to whichever of the goal solved fresh from the start pose and the line's chained answer swings the joints less, not always the flipped branch's answer, so those two rows predate that change). The fixed 0.9 stayed the default until dup picked: **touch is the demo's default since Oct 27** (dup: "the better one"); `run_sort.py` keeps fixed so the PREREG test03/test04 commands rerun as recorded.

Observed in the demo (Oct 23 7:08 PM ET, GPU, viewer, `scripts/demo.py --grip touch --scales --fault wrong_bin`, seed 7, at 834389a; one run, stopped by Ctrl+C after the second part (part_10) landed in rework and before the third (part_12) started, the last sim time in it 38.4 s; log `output/demo/20261023-190856/run.log`, not in git): 0 bump stops; the IK branch-flip fallback fired 3 times (jumps of 0.67, 0.77 and 0.27 rad between 1 cm points); twice the fresh goal swung further than the chained one, so the arm took the chained answer (largest joint swings 1.03 and 0.78 rad), once the fresh one (0.59 rad). The load cells said "no" on the wrong-bin part, the operator rebinned it, and they then said "yes". Pace was 0.15 to 0.40 x real time (event stamps 0.21, 0.17, 0.15, 0.17, 0.15, 0.15, 0.35, 0.40: sim seconds per wall second since the last event; only the two `pick_approach` stretches hold vision calls or the 42 s operator wait (0.21, 0.15). The stretches with no logged wait read 0.15 to 0.17 for part_15's grasp, lift and fault and 0.35 and 0.40 for part_10's grasp and lift; that gap is not explained). None of the three was on a level crossing: two were on `travel`'s straight rise to the 1.03 m clearance height right after a lift, at the pick spot (targets (0.41, 0.16) and (0.43, 0.19), 0.2 s after `pick_lift`; neither tray's wrist target), and the third on the straight descent to the rework tray's hover (target (0.30, 0.60, 0.95), the tray centre plus the grasp offset; the part itself was over the centre). What it shows: flips are common on the vertical legs, not a rare case; the fresh re-solve does not always land nearer. What it does NOT show: how wide those arcs were in metres or whether they came near a tray (not traced), or any more than one seed.

Why that demo ran slow (exploratory, Oct 26, read from logs only, no run; asked by dup: "the last demo i ran seemed to be running really slow ... without running a demo, can you see if there was any obvious cause?"). Method: compared the event pace lines of `output/demo/20261023-190856/run.log` with `output/demo/20261021-153304/run.log` (same seed 7, same parts, same `--fault wrong_bin`); listed the commits between them (`git log --since="2026-10-21 15:30" --until="2026-10-23 19:09"`: only 834389a); read the memory and `nvidia-smi` readings taken around the Oct 23 launch in that session's transcript (kept locally), and the Oct 21 2:05 AM per-8 s `nvidia-smi` log from that session's scratchpad (`demo-resources.log`, not in git). Numbers: the Oct 21 3:33 PM run read 0.43 to 0.59 x real time on the same stretches that read 0.15 to 0.21 on Oct 23 (startup to the first vision call took 9 s vs 27 s). 834389a adds one IK solve per fallback, not per step, and the first slow stretch (0.21, `pick_approach`) came before any fallback fired, so the code is not the cause. **VRAM was over-full:** before the Oct 23 launch the GPU already held 22.4 of 24.6 GB with d3 loaded (d3 alone was about 18 GB when it was stopped at 7:13 PM, so about 4 GB was other apps); on Oct 21 the whole demo with d3 peaked at 22.7 GB from a 3.1 GB start, i.e. Genesis + viewer take about 4 to 5 GB. Oct 23 needed roughly 26 to 27 GB on a 24 GB card; `nvidia-smi` read 23.4 GB mid-run. Native Ollama also loaded embeddinggemma onto the GPU at 7:09 PM, during part_15's slowest stretch. Windows' NVIDIA driver (536.40+, "CUDA - Sysmem Fallback Policy" at its default) spills CUDA allocations into shared system RAM when VRAM runs out instead of failing (NVIDIA support article 5490), which is slow; RAM was tight too (free 8.2 GB, commit 105 GB). What it shows: the likely cause is the GPU memory being full before the demo started, not the code. What it does NOT show: which apps held the extra ~4 GB (not listed at the time), that a spill actually happened (shared-GPU-memory use was not read), or that a demo with the GPU cleared runs at the Oct 21 pace (not run). It also does not explain the 0.42 to 0.49 pace of the Oct 21 runs, which is still the open profiling item.

The dance (same day, dup: "no dancing anymore until i'm happy with everything", "it's dancing into the desk"): off unless `--dance`; when on, the robot raises its arms and reverses until its base is `DANCE_CLEAR_M` (0.8 m) behind the table's near edge. That distance is an estimate from the joint probe's zero pose (hands 0.24 m ahead of the torso, a 19 cm hand, swept round by the turns), not a measured clearance; the fake-sim test checks the base is there before the first beat.

## Exploratory: the robot's own sensors and the grip, Genesis CPU checks (Oct 21; one process per run, `gs.cpu`, decimated robot, cameras off; not GPU, mostly one seed)

What was asked (dup, Oct 21): detect bumps with sensors a real robot has (hand accelerometer vs the MH3's default IMU spots, plus joint torque and a tactile skin), and stop the fingers visibly sinking into the part, with a real sensor if possible. Code: 543ae03 (sensors, bump detection, force grip + grip keeper, fixed closure kept as default) plus the grip options added after it (palm, thumb, lead, substeps, `--close`). Sensor specs and their sources are in the `*Spec` docstrings of `src/handoff/sim/base.py`.

**Sensor smoke** (`scripts/probe_bump_smoke.py`, `scripts/probe_empty_hand.py`, `scripts/probe_skills_bump.py`; rerun Oct 21 at 543ae03 by the port, numbers below are the rerun's):

| Figure | Value | What it shows | Does not show |
|---|---|---|---|
| Hand IMU, clean eased moves (high-passed specific force) | 0.90 to 1.84 m/s^2 | the 3.0 m/s^2 threshold in `config/bump.yaml` clears clean motion (6 sigma of the ZED X noise alone, 1.5, would trip on a lift from rest: first run read 2.15) | real arm vibration; the ZED X "2.3 mg" is read as per-root-Hz, the noisier reading of an unclear spec |
| Slow press onto a pad: hand IMU / joint torque / skin | 6.2 m/s^2 / 33 N*m / 4.4 N (skin at its range) | torque and skin see a slow press; the IMU barely does | anything about real bumps through the skills path (none happened in these runs) |
| Fast crossing into a wall: same | 50 m/s^2 / 34 N*m / 4.4 N | every sensor sees an impact | |
| Clipped vs clean crossings, `probe_bump_sensors.py` seeds 1-2 at 0.25 and 0.40 m/s (re-peaked from the first run's crossings) | clean hand IMU max 1.25; clipped 10 to 15; hand IMU and torque caught 2 of 2 at 10 ms | the detectors separate the two on these 8 crossings | a threshold picked on seeds A and scored on fresh seeds B: that is the GPU run, not done |
| Joint-torque residual std on free motion, armature taken out of the mass matrix | 0.135 to 0.204 N*m per joint (first run 0.12 to 0.17) | matches the 0.164 N*m noise put in; with the full mass matrix joint 2 reads 1.37 | a real arm: the sim's dynamics model is exact, so the 0.5 N*m threshold holds in the sim only |
| Real skill flow, weld grasp, seeds 1-3, 12 guarded moves | hand IMU peak 1.07, torque 0.39; no true contact, the deciding hand IMU never fired | no false stops from the decider | **the skin detector (monitoring only) fired on 4 of 12 clean bin moves, peaks 0.51 to 1.16 N against 0.5**: it must not decide in this flow |
| Empty hand, slow ramp | index and thumb tips touch each other at closure 0.818 (command 0.845), 5.6 to 5.8 N | basis of `EMPTY_CLOSURE` 0.80: a grip force first reached past it is the hand on itself | |
| Empty hand, snap close | every tip saturates at 20 N (thumb on index, three tips on the palm) | a fingertip reading above zero does not mean a part is held | |

**The grip** (`scripts/probe_grip_cpu.py --seed 0 ...`: drive, look, reach, approach, grasp, lift 15 cm, hold 1 s at the top; penetration is the contact solver's deepest hand-link penetration into the part, truth used only to measure):

| Grip (seed 0) | Holds lift + 1 s | Peak penetration | Finger tips after the grasp | Note |
|---|---|---|---|---|
| fixed, close 0.90 (the default until now) | yes | **16.4 mm** (14.3 after the hold) | index 10.5, thumb 13.7 N | the clipping dup sees in the demo |
| fixed 0.90, 2 substeps | yes | 13.3 mm | finger side 38.7 N | stiffer contact (timeconst floor 2*dt) mostly raises the force: substeps are not the fix |
| fixed 0.90, 4 substeps | yes | 12.1 mm | finger side 40.4 N, 72.6 after the hold | same |
| fixed, close 0.85 | yes | 12.2 mm | index 5.9, thumb 5.3 | |
| fixed, close 0.80 | yes | 10.1 mm | index 6.2, thumb 0.9 | |
| **fixed, close 0.75** | **yes** | **5.5 mm** (2.4 after the hold) | index 2.8; palm skin 3.1 N | best so far; multi-seed check below |
| force grip, first version (review run, seeds 0 and 1) | no (seed 0), barely (seed 1) | 3.7 to 4.9 mm tip pads | 14 N, 15x the 0.93 N target | command ran 0.08 ahead of the fingers; the fingers relaxed to the command on the lift |
| force, 0.93 N, lead cap 0.03 + grip keeper | no | 1.1 mm | index 0.8, little 1.2, thumb 0 | keeper tightened 39 times, force fell to 0 within 0.16 s of the lift |
| force, 6 N | grasp fails | 4.0 mm | 5.0 of 6 N after 3 s | the lead cap stops the shared closure before the thumb arrives |
| force 0.93 or 3 N, palm skin >= 1 N required, lead 0.06 | no | 3.3 to 3.5 mm | 6 N on index + little, palm 4.4 N, thumb 0 | palm clamp alone does not hold |
| force 3 N + thumb closed on its own to 5 or 10 N | no | 5.2 to 5.6 mm | finger tips 4 to 5 N, thumb 5 to 7 N | the thumb pushed the part off the palm (palm 4.4 -> 0.3 N); the tips then pressed on the thumb, so the keeper saw a held grip with no part in it |

**Fixed 0.75 vs 0.90 on seeds 1 to 5** (`probe_grip_cpu.py --seed S --grip fixed --close C`, same lift + 1 s hold; seed 0 is the row above):

| Close | Held (seeds 0-5) | Peak penetration, mm (per seed 0..5) | After the hold, mm | Finger side after the hold, N |
|---|---|---|---|---|
| 0.90 | 6 of 6 | 16.4, 16.3, 16.0, 16.4, 15.8, 16.1 | 14.3 to 14.9 | 19.8 to 23.4 |
| **0.75** | **6 of 6** | **5.5, 5.7, 5.5, 5.5, 6.7, 6.8** | 2.2 to 5.2 | 2.6 to 7.1 |

What it shows:
- **Closing to 0.75 instead of 0.9 cuts the sinking about 2.6x (16 to 6 mm peak) and still holds the part on every seed tried.** It is not yet the default: the carry to the bin, the lower and the release were not run, and the GPU contact was not checked (`scripts/probe_grip.py --seeds 20 --modes fixed,fixed@0.75`, then dup's call).
- **A fingertip force sensor cannot tell a part from the robot's own hand.** Every force-grip variant reached its force and still dropped the part on the lift, the last one while every tip read 4 to 7 N. A grip judged by tip force alone is not a grasp check; the fixed grasp is judged by contact (truth) and the lift.
- **What holds is the PD squeeze, not the force reading.** The fixed closures hold because the command stays 0.06 past where the fingers stop; the force grip parks the command next to the fingers.
- **Closing less is the cheap fix for the visible clipping**: 0.75 holds on seed 0 at a third of the penetration.

What it does NOT show: anything on CUDA; more than one seed for most rows; the carry to the bin and the release (only lift + 1 s hold); whether the GPU contact behaves the same. The GPU before/after is `scripts/probe_grip.py`.

## Planned: in_bin from cell sensors, then the camera (methodology, written Oct 19; `scale-in_bin` built Oct 20, not yet scored)

**Status, Oct 20.** The load-cell arm is built: `src/handoff/sensors.py`, switched on with `run_sort.py --scales`. The true load is the Genesis solver's net contact force on each tray's floor and four walls (`Simulator.tray_load`). Fixed in code before any log uses it:
- **Noise model:** a 5 kg straight-bar cell (TAL220B class, listed error ±0.05% of full scale = ±2.5 g, from SparkFun's listing via its forum; the maker's datasheet was not read). Modelled as Gaussian noise with sigma 2.5 g, a 1 g reading step, and each reading the mean over 0.2 s.
- **Rule:** tare both trays before the release, read them after the arm is clear, and compare the gain on each tray with one part's weight (51.2 g).
  - "Yes" when the intended tray gained 0.75 to 1.25 parts and the other tray under 0.25.
  - "No, other bin" when only the other tray gained a part.
  - "No, no bin" when neither tray moved.
  - Anything else is unclear and climbs on to d3, then the person.
- **On the ladder:** the scale is a rung with threshold 1.0 that only in_bin reaches, so a settled reading answers alone.
- **What a "no" does:**
  - A wrong-bin landing goes to the operator, who moves the part. The robot does not re-pick from a tray yet: its grasp is a side grasp of a part standing in the station.
  - A part on the table is re-fed as before.

Checked: Genesis reads a resting part at 0.5023 N (`scripts/probe_tray_load.py`, GPU, config/handoff.yaml) (its mass times g exactly, standard deviation 0 over 50 steps), two parts at 1.0045 N, and empty trays at 0. A 3-episode Genesis smoke run (`run_sort.py --episodes 3 --fault cycle --scales --seed0 9000 --out output\sort\scale-smoke`, the oracle as the person behind the scales) had the scales answer all 9 in_bin decisions, 9 right, none passed up the ladder. The forced drop read "no bin", was re-fed and then read "yes". 16 new tests on the fake simulator, including a forced wrong-bin landing that the scales catch and the operator fixes. A forced wrong-bin run (`run_sort.py --episodes 2 --fault wrong_bin --scales --seed0 9100 --out output\sort\scale-wrongbin`, Oct 21; the fault sets the first part down over the other tray) caught both landings in Genesis. In 9100 a defective part sat in the good tray. Each time the scales read "other bin" and answered "no", the operator moved the part, the next check read "yes", and both episodes sorted 3 of 3 with 8 of 8 in_bin decisions answered by the scales and right.

Scored from the log, Oct 21 (the nightly bug review found `scale_answers` had no caller, so `benchmark.py score` never showed the scale arm): `benchmark.py score --log output\sort\scale-smoke` (seed 0, 30% of episodes held out) now prints a `scale` arm built from the readings logged with each in_bin decision, no replay. On the 21 test decisions it answered the 7 in_bin ones, 7 right, confidently wrong 0, 200 ms each (the reading window). It shows the arm scores from a `--scales` log. It does not show the scales on any ladder: the arm answers in_bin only, so the 95% coverage check keeps it off every ladder, and it is not part of the frozen ladder. Three episodes is a smoke run, not a measurement.

Not yet: the beam, robot and fused arms; the PREREG section; test05; a robot re-pick out of a tray.

Why: from one frame the camera model does not catch a wrong-bin landing when the target tray already holds a part (above), and the hand-off rate on clean runs is too high for a line that is meant to run without an operator beside it (dup, Oct 19). Professional cells confirm a physical state with a physical sensor and keep vision for what sensors cannot see; Boston Dynamics describes Stretch judging its grip with pressure and optical sensors in the suction gripper and using its 2D/3D perception mast to spot boxes that fell, then recovering them itself (bostondynamics.com, robotsguide.com; their exact drop-detection logic is not published).

Sensors to model, one arm each (simulated from Genesis state, each with a stated noise model; none is a real device):

| arm | sensor | in_bin rule | noise model to apply |
|---|---|---|---|
| `scale-in_bin` | a load cell under each tray | yes when the intended tray's weight rose by about one part's mass since the release and the other tray's did not | Gaussian reading noise and a resolution step sized to a typical bench load cell (values chosen and written down before the run) |
| `beam-in_bin` | a through-beam light curtain across each tray's mouth | yes when the intended tray's beam broke after the release and the other's did not | a stated miss probability per crossing |
| `robot-in_bin` | the robot's own record: release pose over the intended tray, gripper reports empty afterwards | yes when both hold | none (it is the controller's own data); cannot see a bounce into the other tray, which is why it is not used alone |
| `fused-in_bin` | scale + beam, camera only on disagreement | the two sensors agree: their answer; they disagree: ask d3; d3 under threshold: person | from the parts above |

Evaluation, so the result can be checked:
1. Write the sensor noise values, the fusion rule, and the four rates to report (acted wrong, confidently wrong, handed to a person, alone and right; plus wrong-bin landings caught) into a PREREG section and commit it before the log is made.
2. Make a new log, `test05`, the same way as test04 (300 episodes, new seeds, fault cycle). test03 and test04 are not reused for this claim: test04 is spent, and test03 chose the ladder.
3. Replay d3 on test05 at the frozen threshold, compute each sensor arm from the logged state, score once with by-episode 95% intervals beside d3 alone.
4. Report per question and per fault type with `scripts/handoff_breakdown.py`, as above.

The same reasoning applies to sort_complete (the largest source of clean-run hand-offs): "is every part in a bin" is a count over the per-tray sensors plus the table camera, not a camera judgement alone. In scope for the same test05 run if built.

## Five more decision models alone on `test03`, scored Oct 17 (`benchmark.py score --log output/sort/test03` at 13af40a; same split and thresholds method as the Oct 14 table; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong |
|---|---|---|---|---|---|---|
| clef-flash 9B, frame | 0.840 | 0.047 | 0.085 | 0.88 | 0.854 | 0.069 |
| nimble-v3 9B, text + Gemma description | 0.584 | 0.377 | 0.386 | 0.97 | 0.484 | 0.463 |
| tev1 4B, text + description | 0.555 | 0.383 | 0.394 | 0.95 | 0.441 | 0.469 |
| tev1 0.8B, text + description | 0.515 | 0.290 | 0.347 | 0.80 | 0.383 | 0.420 |
| d1 3B, frame | 0.488 | 0.000 | 0.336 | 0.66 | 0.345 | 0.000 |

Served by the native Windows Ollama 0.40.4-rc1 decision endpoint (`/v1/systemone`), 1,238 answers per arm (test03 and log02b), no errors; median 27 to 135 ms per answer.

What it shows: "decision models are calibrated" is not true of the class. nimble-v3 and tev1 4B are confidently wrong on 38% of decisions, as bad as Gemma, while reading the same state text and description on which the three text Deciders stay under 4%. clef-flash is the most accurate single arm yet (0.84) but confidently wrong 4.7%. d1 is never confidently wrong because it is rarely confident (mean 0.66) and right under half the time. Calibration is a property of particular models, which is why it has to be measured per arm.

What it does NOT show: no intervals in this row (they are in `benchmark.json`); one split (seed 0); ladders with these arms are not scored yet; the text arms depend on Gemma's description of the frame.

## d3-nano and Intern-Decision-4B alone on `test03`, scored Oct 17 (`benchmark.py score --log output/sort/test03` at a94b38c; same split and method as above; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong |
|---|---|---|---|---|---|---|
| Intern-Decision-4B, frame (rerun Oct 18, fixed server) | 0.849 | 0.003 | 0.050 | 0.81 | 0.867 | 0.004 |
| d3-nano 2.2B, frame | 0.802 | 0.012 | 0.068 | 0.85 | 0.799 | 0.017 |
| (for reference) Decider 2B, frame | 0.780 | 0.000 | 0.078 | 0.72 | 0.767 | 0.000 |
| (for reference) pplx-decider 27B, frame | 0.805 | 0.016 | 0.068 | 0.87 | 0.803 | 0.024 |
| (for reference) clef-flash 9B, frame | 0.840 | 0.047 | 0.085 | 0.88 | 0.854 | 0.069 |

Served on native Windows: d3-nano by its own `d3_server.py` (transformers 5.17, own kernels; 150 to 500 ms per answer), Intern-Decision-4B by `scripts/intern_server.py` around its `DecisionEngine` (transformers 5.14.1; the recorded ~9 s per answer is NOT the model's: torch 2.9.1's bundled cuDNN 9.10 sent the vision patch embedding's bf16 Conv3d to a fallback that spent ~8.7 s of each ~8.8 s forward pass (profiled 2026-10-18). With that Conv3d run as the equivalent matmul, plus causal-conv1d and flash-linear-attention installed, the same server answers in a median 151 ms (40 test03 decisions, same 40/40 top answers as the recorded rows, max probability change 0.017; the fast kernels alone account for 504 to 132 ms). Rerun Oct 18 with that server on both logs (d11fbb4): 1,238 answers, no errors, median 152 ms; 9 top answers changed (1 log02b, 8 test03), largest probability change 0.19; the Oct 17 row was 0.848 right / 0.006 confidently wrong / ECE 0.043; every other arm's score is unchanged). 1,238 answers each, no errors left.

What it shows: Intern-Decision-4B is the most accurate arm so far and the best calibrated (ECE 0.050), confidently wrong on 0.3%, at about 150 ms an answer on the 3090. d3-nano at 2.2B matches the 27B. Ladder choice is still to be made on held-out episodes only (docs/PREREG.md).

What it does NOT show: no intervals here (in `benchmark.json`); one split; which patch-embed path is closer to full precision (no fp32 reference run). Checked Oct 18 on the 9 flips plus the 2 largest shifts, each setup twice: every answer repeats exactly (no run-to-run noise), and the old and new arm rows reproduce exactly. The patch-embed swap alone moves all 9 flips and the 0.19 shift (log02b 41 in_bin, 0.72 to 0.53, right either way); the fast kernels alone move answers by one or two bf16 steps (~0.016 each). All 9 flips were 0.41 to 0.53 before and after. Scripts and results: `D:\Toolkits\build\profile\flip_*`.

## Six new frame arms on `test03`, scored Oct 18 (`benchmark.py score --log output/sort/test03` at eb1c1a1; same split and method as above; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong | right, by-episode interval | median ms |
|---|---|---|---|---|---|---|---|---|
| Intern-Decision-2B, frame | 0.848 | 0.000 | 0.167 | 0.68 | 0.865 | 0.000 | 0.80 to 0.89 | 164 |
| d3-mini 4B, frame | 0.845 | 0.074 | 0.109 | 0.93 | 0.861 | 0.107 | 0.80 to 0.89 | 245 |
| pplx-decider-flash-v1.2 4B, frame | 0.831 | 0.009 | 0.070 | 0.86 | 0.842 | 0.013 | 0.78 to 0.88 | 209 |
| d1-omni 600M, frame | 0.475 | 0.123 | 0.362 | 0.84 | 0.325 | 0.092 | 0.45 to 0.50 | 74 |
| d3-edge 0.6B, frame | 0.469 | 0.435 | 0.489 | 0.85 | 0.317 | 0.630 | 0.45 to 0.49 | 84 |
| Winnow-12B Q8, frame (scored at 6c930e1) | 0.825 | 0.015 | 0.019 | 0.82 | 0.833 | 0.021 | 0.78 to 0.86 | 299 |
| (for reference) Intern-Decision-4B, frame | 0.849 | 0.003 | 0.050 | 0.81 | 0.867 | 0.004 | 0.80 to 0.89 | 155 |

test03 only (log02b is legacy and no choice is made on it; Claude's default, dup agreed Oct 18). 974 answers per arm, no errors. Served on native Windows: d3-mini and d3-edge by each package's own `d3_server.py` in the d3 venv (port 8768; reference PyTorch path, flash-linear-attention not installed there), Intern-2B by `scripts/intern_server.py` with the patch-embed matmul (port 8767), pplx-flash-v1.2 by the new `scripts/pplx_server.py` in its own venv with the model's pins (torch 2.14.0, transformers 5.17.0; port 8769; one question per request, the card's recommended default), d1-omni by Ollama 0.40 (`d1-omni`, 50-answer chunks, unloaded after each). Weights in `D:\Models\hf-win` (pulled Oct 18, dup's yes).

What it shows: the top five frame arms (Intern 4B and 2B, d3-mini, clef-flash, pplx-flash-v1.2) are within one interval of each other on accuracy (0.83 to 0.85), so they separate on calibration, not accuracy. Intern-2B is never confidently wrong because it is underconfident (mean confidence 0.68 at 0.85 right, ECE 0.167); d3-mini is right as often but confidently wrong on 7.4%. pplx-flash-v1.2 (0.9% confidently wrong, ECE 0.070) beats its own 27B v1.1 Artemis GGUF here at a third of the latency. d3-edge and d1-omni are below 0.5 right and not candidates.

What it does NOT show: one split (seed 0); no ladders with these arms yet; underconfidence (Intern-2B) may be fixable by the ladder's thresholds, which this table does not apply. Winnow-12B ran in WSL by dup's exception (Oct 18): EldanRing winnow-inference at a50c7d0 built for sm_86 against a CUDA 13.2 redist toolkit, Ollama's Q8 and mmproj files reused (sha256 match the release manifest), 8K context, vision on, reasoning and MTP off, temperature 1 (the server's default), `--load-mode dio`. Every load added about 23 GB of Windows commit (WSL file read plus what looks like system backing for WSL's GPU allocations; mechanism inferred from readings, not documented), peaking at 99.5 of 128 GB, so the 88 GB guard was lifted to 100 for this run only (dup's yes). Winnow has the lowest ECE on the board (0.019): its stated confidence matches how often it is right, at 0.83 right and 1.5% confidently wrong.

## Nine more frame arms on `test03`, scored Oct 18 (`benchmark.py score --log output/sort/test03` at cc7c2fd; same split and method; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong | right, by-episode interval | median ms |
|---|---|---|---|---|---|---|---|---|
| Imajev-4B (LoRA on Qwen3.5-4B), frame | 0.858 | 0.012 | 0.037 | 0.87 | 0.880 | 0.017 | 0.81 to 0.90 | 912 |
| JPT-4B, frame | 0.855 | 0.009 | 0.084 | 0.77 | 0.876 | 0.013 | 0.81 to 0.90 | 228 |
| d3-flash 9B, frame | 0.848 | 0.010 | 0.086 | 0.86 | 0.865 | 0.015 | 0.80 to 0.89 | 269 |
| RSI-Jev v6.1 VL 4B, frame | 0.845 | 0.049 | 0.116 | 0.80 | 0.861 | 0.071 | 0.80 to 0.89 | 213 |
| basal-omni-1.0-mini (Gemma 4 E4B), frame | 0.794 | 0.090 | 0.115 | 0.91 | 0.788 | 0.045 | 0.77 to 0.82 | 296 |
| Imajev-9B (LoRA on Qwen3.5-9B), frame | 0.771 | 0.000 | 0.096 | 0.79 | 0.754 | 0.000 | 0.72 to 0.82 | 290 |
| JPT-9B, frame | 0.768 | 0.016 | 0.135 | 0.81 | 0.749 | 0.024 | 0.73 to 0.81 | 246 |
| Intern-Decision-0.8B, frame | 0.602 | 0.000 | 0.275 | 0.57 | 0.510 | 0.000 | 0.57 to 0.64 | 152 |
| d3-lite 0.8B, frame | 0.580 | 0.049 | 0.249 | 0.77 | 0.478 | 0.071 | 0.55 to 0.61 | 183 |
| (for reference) Intern-Decision-4B, frame | 0.849 | 0.003 | 0.050 | 0.81 | 0.867 | 0.004 | 0.80 to 0.89 | 155 |

974 answers per arm, no errors left (RSI-Jev's first pass returned 422 on every request because its server requires a `model` field; those rows are errored, ignored by scoring, and the arm was rerun with `model: rsi-jev-v6.1-vl-4b`). Weights pulled Oct 18 into `D:\Models\hf-win` (dup's yes, ~104 GB). Each server in its own native Windows venv under `D:\Models\decision-servers\` (setup.ps1 beside each), at the maker's settings: d3-flash / d3-lite by each package's `d3_server.py` (T 1.0); Intern-0.8B by `scripts/intern_server.py` (patch-embed matmul; its `inference.py` temperature); RSI-Jev by `rsi-jev serve` (torch 2.14.0, transformers 5.17.0, no `[fast]`; T 1.02 from its calibration file); Imajev by the repo's `scripts/playground/server.py` on the pinned Qwen3.5 base revisions, 4B with `--rotations 4` and `calibration-rot4.json` (the card's command; hence ~0.9 s), 9B with `calibration.json` (its card's command); JPT by `llm2jev --backend hf` with the card's temperatures (4B 1.036, 9B 1.087), the frame inside `state` as a chat message; basal-omni-mini by `basal-serve --mode eager --media on` (torch 2.11, transformers 5.17; T 1.139 for two-option choice, fitted on text only), the frame in `media`. JPT is CC BY-NC 4.0 (fine for a benchmark row, not for shipping). Windows commit peaked at 99.2 GB on load (9B arms); dup set a 110 GB cap for this batch only, watchdog never fired. d3-flash ran before the live watchdog existed and reached 106.9 GB.

What it shows: two new arms, Imajev-4B (0.858) and JPT-4B (0.855), top the accuracy column, but every arm from 0.840 to 0.858 is inside one by-episode interval of the others: the top group is now nine arms, separated by calibration and confidently-wrong rate, not accuracy. Among them Intern-4B (0.3%), JPT-4B (0.9%), d3-flash (1.0%) and Imajev-4B (1.2%) stay near zero confidently wrong; RSI-Jev (4.9%) does not. Both 9B LoRAs score below their own 4B versions here (Imajev 0.771 vs 0.858, JPT 0.768 vs 0.855). The 0.8B models are not candidates (0.58 to 0.60).

What it does NOT show: one split (seed 0); ladders with these arms not scored; why the 9B versions trail their 4B versions (not investigated; Imajev-9B ran without option rotations, Imajev-4B with four).

## 27B frame arms as 4-bit GGUFs on `test03`, scored Oct 18 (`benchmark.py score --log output/sort/test03` at f447dd6; same split and method; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong | right, by-episode interval | median ms |
|---|---|---|---|---|---|---|---|---|
| d3 27B, Artemis IQ4_XS GGUF, frame | 0.898 | 0.006 | 0.037 | 0.88 | 0.938 | 0.009 | 0.86 to 0.93 | 277 |
| Clef 27B, Artemis IQ4_XS GGUF, frame | 0.862 | 0.004 | 0.102 | 0.81 | 0.872 | 0.006 | 0.82 to 0.90 | 302 |
| (for reference) Imajev-4B, frame | 0.858 | 0.012 | 0.037 | 0.87 | 0.880 | 0.017 | 0.81 to 0.90 | 912 |
| (for reference) Intern-Decision-4B, frame | 0.849 | 0.003 | 0.050 | 0.81 | 0.867 | 0.004 | 0.80 to 0.89 | 155 |

974 answers per arm, no errors. TurinTech's Artemis IQ4_XS files (`turintech/d3-27B-Artemis-GGUF`, 14.9 GB; `turintech/Clef-Artemis-GGUF`, 14.8 GB) with ggml-org's BF16 vision projectors (`ggml-org/vllm-d3-GGUF`, `ggml-org/Clef-GGUF`), served by llama.cpp b11723 (Windows CUDA 13.4 release build, `D:\Models\llama.cpp\b11723`) at the cards' settings: `-c 16384 -ub 8192 -b 8192 -np 1` (one slot: the card notes batching can change a d3 answer), all layers on the GPU, `/v1/systemone` with data-URL images. Temperature: whatever the GGUF carries (not overridden). Windows commit at load: 105.0 (d3) and 99.2 GB (Clef) under dup's 110 cap; at the normal 88 the watchdog killed the first d3 load at 93.4.

What it shows: d3 27B is the first arm clearly ahead of the pack: 0.898 right (interval 0.86 to 0.93), 0.938 on judgement questions, 0.6% confidently wrong, ECE 0.037, at 277 ms, a 4-bit quant on one 3090. Its interval still touches the top of the 4B group's (up to 0.90). Clef 27B (0.862, 0.4% confidently wrong) sits at the top of the 4B group, not above it, and its confidence runs low (ECE 0.102).

What it does NOT show: one split (seed 0); the full-precision d3 27B (54 GB, does not fit); how much the 4-bit quant costs (TurinTech's card has its own panel, not our frames); whether JEV-27B-VL, RSI-Jev 27B or pplx v1.2 27B would do as well (JEV-27B-VL GGUF needs a patched llama.cpp; the other two have no quant that fits the 3090 off Blackwell).

## Tier-1 frame arms through their makers' engines on `test03`, scored Oct 19 (`benchmark.py score --log output/sort/test03` at e6eab19; same split and method; 676 test decisions)

| arm | right | confidently wrong | ECE | mean confidence | judgement right | judgement confidently wrong | right, by-episode interval | median ms |
|---|---|---|---|---|---|---|---|---|
| OneJev 1.6 27B, IQ4_XS GGUF, frame | 0.870 | 0.003 | 0.044 | 0.84 | 0.897 | 0.004 | 0.83 to 0.91 | 678 |
| StartLux-Decision v1.1 27B Instant, Q4_K_M GGUF, frame | 0.854 | 0.003 | 0.085 | 0.83 | 0.874 | 0.004 | 0.82 to 0.89 | 362 |
| Amos 1 medium, frame | 0.777 | 0.009 | 0.125 | 0.86 | 0.762 | 0.013 | 0.73 to 0.82 | 263 |
| OneJev 1.6 9B, Q4_K_M GGUF, frame | 0.769 | 0.006 | 0.156 | 0.79 | 0.754 | 0.009 | 0.72 to 0.81 | 302 |
| (for reference) d3 27B, Artemis IQ4_XS GGUF, frame | 0.898 | 0.006 | 0.038 | 0.88 | 0.938 | 0.009 | 0.86 to 0.93 | 277 |

974 answers per arm, no errors. These GGUFs carry no `<arch>.decision.*` metadata, so llama.cpp b11723 alone refuses them ("This model is not a decision model"); each ran behind its maker's own engine. StartLux: `startlux-decide serve --backend llamacpp` (StartLuxLabs/StartLux-Decision, tokenizer and `decision_config.json` from the BF16 repo) in front of llama-server with the card's image flags (`--image-min-tokens 64 --image-max-tokens 1024 -fa on --ctx-checkpoints 0 --cache-ram 0`) but `-c 16384 -ub 8192` instead of the card's 20480 / 16384 (that load reached 113 GB commit); stock llama.cpp without the maker's patch 0003, so text and image go in separate micro-batches, which the maker says leaves read-outs about 4 times further from BF16 (0.022 against 0.0055 mean on their 4B). OneJev: `qev serve --gguf` (OmniJev/OneJev), which starts llama-server itself with at least 64 image tokens; the frame goes in `media` and the state names it `<image:1>` (checked: one probe with and without the frame moved "good" from 0.56 to 0.98). Amos: the model repo's own `serve/` package (`python -m amos.serve --rotations 1 --strict --no-state-cache`, the Decision Index settings; torch 2.14.0 cu130, transformers 5.17.0); transformers did not pick up flash-linear-attention and ran its pure-PyTorch linear attention, which the maker says can move probabilities by up to 0.02. Windows commit at load: 105.0 (StartLux), 93.7 (OneJev 27B), 95.3 (Amos), 83.3 GB (OneJev 9B), under the overnight 110 cap.

What it shows: OneJev 27B is second on the board (0.870, 0.897 on judgement questions, 0.3% confidently wrong, ECE 0.044), behind d3 27B; their intervals overlap. StartLux 27B sits at the top of the 4B group. OneJev 9B and Amos 1 medium are below the 4B group on our frames: OneJev 9B answers `sort_complete` "yes" every time and `finish` "defective" 89% of the time.

What it does NOT show: one split (seed 0); full-precision weights; StartLux with the maker's llama.cpp patch. JEV-35B (engine is vLLM, one GPU with 80 GB or more) and KnowLine-35B (SGLang FP8, 48 GB or more) were not run.

## How many arms were replayed on `test03` (counted Oct 27, for the writeup)

40 distinct arm names in `output/sort/test03/arms.jsonl`, counted with `python3 -c "import json,collections; c=collections.Counter(json.loads(l).get('arm') for l in open('output/sort/test03/arms.jsonl')); print(len(c))"`. Two are cloud (`argon-vision`, `gemini-er2-vision`); the other 38 ran locally on the 3090. Does not show: that every arm was scored on all 974 decisions (the cloud ones answered a sample; see their sections), or results for `gemini-er2-vision`, which has no scored section here.

## The notebook's recomputation of test04, and the test count (Oct 27)

`notebooks/decision_ladder_results.ipynb` recomputes the test04 confirmation from `notebooks/data/test04_decisions.csv` (an extract of `output/sort/test04/decisions.jsonl` and `arms.jsonl`, regenerated byte-identical by `scripts/export_notebook_data.py`, md5 c88ca55e), and its outputs were saved by `scripts/execute_notebook.py` (nbconvert is not installed; that script runs the code cells through `run.ps1`). Every rate and interval it prints equals `confirmation.json` and `live-equivalent.json` to 4 decimals, as do the per-question, per-fault and wrong-bin counts. Checked by the agent that built it and spot-checked in the main session (the 0.0115 / 0.19245 / 0.0915 rounding and the 29 decisions on 15 landings, from the raw files). Does not show: anything not already in the confirmation; the plots are illustrations, not refits.

Test suite at that commit: **322 passed, 16 skipped** (`.\run.ps1 pytest -q`, the whole suite: the project's tests on a fake simulator plus the starter's; none needs a GPU).

## Temperature per arm (recorded Oct 18; applies to every arm row above)

Rule: every decision-model arm runs at its maker's own temperature, never ours. Each maker fits that value on its own data so the model's probabilities come out calibrated; one shared number would mis-calibrate most of them. Decision models generate no text, so temperature changes only how spread out the option probabilities are, never the top answer. Every question in these logs is sent as type `choice`, so per-type temperatures resolve to the choice value.

| arm | temperature | source |
|---|---|---|
| Intern-Decision-4B | 1.992 | `DEFAULT_TEMPERATURE` in the snapshot's `inference.py` (applied by `DecisionEngine`) |
| Intern-Decision-2B | 2.101 | same, 2B snapshot |
| pplx-decider-flash-v1.2 | 1.554 | snapshot `decision_config.json` |
| d3-nano, d3-mini, d3-edge | 1.0 | each package's `decision_config.json` |
| Decider 2B (frame and text) | 0.863 (choice; global 1.055) | `strands_decider_config.json` `temperature_by_kind` |
| Decider E2B text | 0.561 (choice; global 0.800) | same |
| Decider E4B text | 0.775 (choice; global 0.818) | same |
| Winnow-12B | 1.0 | winnow-inference server default (docs/API.md: other values need held-out calibration) |
| d3-flash, d3-lite | 1.0 | each package's `decision_config.json` |
| Intern-Decision-0.8B | `DEFAULT_TEMPERATURE` in its `inference.py` | same mechanism as 4B / 2B |
| RSI-Jev v6.1 VL 4B | 1.02 | its `calibration.json` (per the card) |
| Imajev-4B | rotation-4 fit | `calibration-rot4.json` (card command, `--rotations 4`) |
| Imajev-9B | 1.748 (per the card) | `calibration.json` |
| JPT-4B / JPT-9B | 1.036 / 1.087 | card's `llm2jev --temperature` value (nothing stored in the repo) |
| basal-omni-mini | 1.139 (two-option choice) | its `CALIBRATION.json`, fitted on text only |
| d3 27B, Clef 27B (Artemis GGUFs) | as stored in the GGUF | llama.cpp's systemone readout; not overridden, not inspected |
| StartLux-Decision v1.1 27B Instant | 1.0867 (choice; noul 1.7238, score 1.1841) | BF16 repo's `decision_config.json` `temperature_by_type`, read by `startlux-decide` |
| OneJev 1.6 9B / 27B | 1.0 (none shipped) | qev reads `calibration.json` from the model repo; OmniJev's repos have none |
| Amos 1 medium | 1.0 (none fitted) | its `calibration.json` holds only quote (`span`) settings, no temperature |
| clef-flash, d1-3B, d1-omni, nimble-v3, tev1, pplx Artemis (Ollama) | not exposed | inside each model's Ollama decision endpoint; `ollama show` lists no temperature parameter (checked for clef-flash and d1:3b) |
| Gemma one-word and self-confidence arms | 0 (text generation) | `src/handoff/rungs.py` request options; makes the generated word or JSON repeatable. Not a probability rescale: one-word probabilities come from the first token's log-probabilities, self-confidence ones from the model's stated number |
| Argon, Robotics ER 2 (cloud) | provider default | stated confidence in JSON; no log-probabilities |

None of these were overridden by the benchmark. Not checked: whether Ollama applies a temperature inside its decision endpoints at all.

## Decider server moved from WSL to native Windows, parity check (Oct 17)

| Figure | Value | Command | Does not show |
|---|---|---|---|
| Same top answer as the recorded WSL answers | 39 of 40 test03 decisions (random sample, seed 7); largest probability difference 0.017 | `D:\Models\decision-servers\decider\serve.cmd` (strands-decider at 0ac22a9, torch 2.14.1, transformers 5.19.0, weights from the hard-link mirror `D:\Models\hf-win`), then a 40-decision comparison script | The one flip is a near-tie (difference at most 0.017), the size of GPU numeric noise between builds; recorded arms were not re-run, so every Decider number above stays as measured under WSL. |

## Spoken operator answers (Nov 2; a plumbing check, not an accuracy number)

Question: does a spoken reply reach the right option, with the person confirming, and does a failure leave the buttons working?

| check | method | result | what it does NOT show |
|---|---|---|---|
| Reading typed replies | `.\run.ps1 python scripts/probe_voice_map.py` (d3 27B; `output/voice_probe/map.json`) | 41 of 41 as labelled: 34 proposed the right option, 7 that pick none were asked again; 0 wrong | Real speech. I wrote all 41 phrases and their labels myself, before seeing any answer, and they are mostly clean |
| Whole path in a browser | `.\run.ps1 --with playwright python scripts/probe_voice_page.py <clip> <option> <question>` (Edge, fake mic, real Whistle, real d3) | 2 of 2 clips reached the right option and were logged as spoken after a confirm; about 0.5 s from stopping to the proposal; the question stayed open until the confirm | A real voice, mic or room: the clips are Windows' speech voice |
| Whistle on Windows | `needle.exe --model whistle.cact --audio <clip>` on 5 such clips | About 110 to 240 ms a clip, CPU only. Close, with small slips ("look scratched", "sended to rework") | Accuracy on people. Whistle's published error rates are its maker's, measured on an M4 Pro |
| Awkward clicks | `.\run.ps1 --with playwright python scripts/probe_voice_races.py` | Failed before the fix (2 microphones open on a double click), passes after | Other browsers |

Not run: a real microphone, a full `demo.py` with Genesis, `--ladder local` with the Decider 2B.

## Capsule tray sizing: how many soft gels Genesis carries (Nov 4; a sizing probe, not a result)

Question (dup, Nov 4): real trays hold hundreds of capsules; how far can the sim go? Method: `.\run.ps1 scripts/capsule_count_probe.py --n <N> [--free <k>] [--height <m>] --frame <png>`: the kit's robot parked out of the way, a 45 x 35 cm tray with matte parchment, N capsule meshes (24 x 11 mm) on a 30 x 16 mm grid, a visual-only pool under every tenth; 50 warm-up steps, then 300 timed steps at dt 0.01 on the RTX 3090 (2.6 GB in use before, nothing else on the GPU). Rows in `output/bench/capsule_count.jsonl`, frames beside it.

**Correction (same night):** the first runs started every capsule 0.5 mm inside the parchment (the probe took the paper's centre for its top; found by the commit review). Their drift included that push-out: a "12.7 mm, round pills roll" row led to flattened pills, and it was wrong. Those rows are kept in `capsule_count_before_paper_fix.jsonl`; the rows below are the rerun.

| capsules | cross-section | loose | sim speed | build | largest drift at rest (median) |
|---|---|---|---|---|---|
| 35 | round, 11 mm | 35 | 1.19x real time | 44 s | 1.7 mm (1.4) |
| 35 | flattened, 8.5 mm tall | 35 | 1.26x | 35 s | 1.6 mm (0.8) |
| 140 | flattened | 140 | 0.72x | 37 s | 2.4 mm (0.7) |
| 266 (full grid) | flattened | 266 | did not run before the fix: Genesis exceeded its 750 candidate contact points (not rerun) | | |
| 266 | flattened | 10 (the rest fixed in place) | 1.32x | 46 s | 1.7 mm (0.8) |

What it shows: a full tray runs faster than real time when only the capsules the robot will move are loose bodies; every loose capsule costs speed; all 266 loose did not run at the default contact budget. Round pills sit as still as flattened ones, so the cells use round ones. What it does NOT show: the robot working the tray, a dump of the tray (fixed capsules cannot fall, so a physical dump of hundreds is not available at this budget), speed with the viewer open, or run-to-run spread (the same 35-capsule case read 1.04x and 1.26x on two runs).

## The MH3's reach and size, for the capsule cell (Nov 4; geometry probes, not results)

Question: where can the robot work a 45 x 35 cm tray? Method: `.\run.ps1 scripts/probe_body.py` (every link's collision box at the kit's home pose, relative to the base origin) and `.\run.ps1 scripts/probe_reach.py --z 0.88` (each arm sent to a grid of wrist targets 0.25 to 0.75 m ahead, 0.6 m either side, step 0.05 by 0.10, in the grasp orientation, relative to the base; `#` within 1 cm). Files: `output/bench/body.json`, `output/bench/reach_z088.json`.

- **Body:** mobile base x -0.34 to +0.34, y +-0.29, z 0.12 to 0.41 m; lift column x -0.14 to +0.10; torso x -0.10 to +0.14, z 0.94 to 1.29; shoulders at about x 0, y +-0.13 to 0.26, z 1.2.
- **Left wrist reach at z 0.88 (right mirrors it):** x 0.25 to 0.30 for y 0.1 to 0.6; x 0.35 to 0.40 for y 0.1 to 0.5; x 0.45 for y 0.1 to 0.4; x 0.50 for y 0.2 to 0.3; nothing from 0.55 on, and nothing at y 0 (the midline). Hit points were 2 to 7 mm off.

What it means for the cell: against a solid table (base front 0.34 m ahead) the tray would sit past the hand's band, so the cell table is a top on legs and the base rolls under it until the torso is 2 cm from the edge; the left hand then covers a band 0.37 to 0.52 m ahead of the base, and the robot drives sideways between leakers (`capsule_cell.park_for`, tested over every grid spot). What it does NOT show: reach below z 0.88, with the lift column moved, or with the hand turned; whether driving under the table top is clean in Genesis (not run yet).

## Driving between the cells and under a table top (Nov 4; a skill check, not a result)

Method: `.\run.ps1 scripts/probe_drive_cells.py` (Genesis, arms in the travel pose): from the aisle spawn, tuck in under cell a's table at the base spots `park_for` gives for four points (tray near-left, far-right, middle, the leaker bin), shifting between them, then back out and park at cell b. Errors are the base origin's distance and heading from the goal, by `base_pose`. Log `output/cells/drive/drive.json`, a frame per stop beside it.

- **First run, steering the base origin:** every park under cell a missed by 76 to 203 mm and the robot stopped turned off to the side. The MH3 turns about its drive-wheel axle, 0.20 m ahead of the origin, so each turn in place swung the origin sideways. The fake simulator turned about the origin, so its tests could not see this; it now turns about the same axle.
- **Steering the axle point:** all six moves within 10.5 to 21.0 mm and 1.3 to 1.7 degrees, 12 to 16 s of sim time each. Backing out still turned 16 degrees; reversing out and the last stretch in now hold their heading (fake-tested; not yet rerun in Genesis).

What it does NOT show: arms anywhere but the travel pose under the table, or bumps (no contact was read).

## Picking leakers off the tray, in Genesis (Nov 4; a skill check, not a result)

Method: `.\run.ps1 scripts/probe_pick_leaker.py --spots 130,9,250,0,259` (cell a: the tray's middle and four corners). Each leaker is loose with its pool, the capsules around it fixed. Per leaker: park for it (`park_for`, `tuck_in`/`shift_under`), weld-pick it (hand down to 4 mm over it, weld, lift 10 cm), park for the leaker bin, drop it from 1 cm above the rim. Whether it ended in the bin is read from the simulator's truth, for this check only. Log `output/cells/pick/pick.json`, frames beside it.

| run | change before it | in the bin | notes |
|---|---|---|---|
| 1 | | 1 of 3 | two parks 40 and 58 mm off: the last stretch in held its heading exactly, so the approach spot's sideways miss stayed |
| 2 | last stretch may steer 6 degrees off the heading; a park counts up to 7 cm off (the hand's band has room) | 2 of 3 | every step reported ok; one capsule fell through the leaker bin's 1 cm floor into the table (14 mm a physics step at that speed) |
| 3 | leaker bin floor 3 cm; let go 1 cm above the rim | **5 of 5** | parks 16.5 to 26.5 mm off, hand over the capsule 0.9 to 3.3 mm off, about 37 s of sim time per leaker |

What it does NOT show: cell b in Genesis (fake-tested only), leakers next to each other, a capsule bumped by the hand on the way down (the neighbours are fixed, so they cannot be), or any decision: the leakers were named, not found.

## Capsule selection log sel01 (Nov 4-5; the simulator's truth answering, for replaying models on)

Method: `.\run.ps1 scripts/run_capsules.py --episodes 150 --seed0 1000 --out output/capsules/sel01` at 541877e (the shift tree, five questions, block size 16, lit 1920 x 1200 tray camera, leakers fixed like every capsule with a pool under them, capsules nudged up to 0.8 mm and 8 degrees). Oracle ladder: every answer is the truth, so this log measures the robot's moves and the labels, not any model. Files: `output/capsules/sel01/{episodes,decisions}.jsonl`, `frames/`.

- **Shifts:** 150 of 150 finished; work in cell a 72, cell b 69, neither 9. 3.95 h of wall time (about 1.6 min a shift).
- **Leakers:** 282 placed, 282 in the leaker bin, 0 left on a tray, 0 good capsules thrown out.
- **Perception:** 260 of 260 capsules found in each of the 141 tray scans.
- **Decisions logged:** 11,034: leak_capsule 7,500, leak_block 2,961, leaker_landed 282, where_needed 150, tray_clear 141.

What it does NOT show: how any model answers (the replays come next), the tray dump (not built), the cart slide (not built), or speed with a viewer open. The block-size curve was cut from the first 20 scans (`capsule_block_curve.py --max-scans 20`).

## First replays on the capsule selection log (Nov 5 morning; EXPLORATORY: selection data, nothing frozen)

Method: `scripts/replay_capsules.py` (one model on the GPU at a time, VRAM and commit gated) replayed d3 27B, Qwen3.8-27B (first-word probability and stated confidence), Intern-Decision-4B and Decider 2B on `output/capsules/sel01` (11,034 decisions) and on the block-size cuts of its first 20 trays. Counts by `python3 scripts/capsule_catch_rates.py <log>`: a leaker picture is leak_block "yes", leak_capsule "leaker" or tray_clear "no".

| leakers caught (missed at 0.9 or more) | d3 27B (VDM) | Intern 4B (VDM) | Qwen3.8-27B first word (VLM) | Qwen3.8-27B stated (VLM) |
|---|---|---|---|---|
| the live run's questions, 828 leaker pictures | 74% (121) | 85% (66) | 35% (275) | 55% (370) |
| one capsule a picture, 38 leakers | **100% (0)** | 100% (0) | 26% (8) | partial run |
| 4 a picture, 120 | 62% (27) | 66% (24) | 32% (56) | not run yet |
| 16 a picture, 77 | 51% (22) | 70% (14) | 39% (35) | not run yet |
| 64 a picture, 47 | 68% (8) | 79% (6) | 60% (14) | not run yet |
| whole tray, 17 | 82% (1) | 100% (0) | 76% (1) | not run yet |

False alarms on clean pictures were low for d3 (0.7% on the live questions, 0 on single capsules) and Qwen (2.0%); Intern 6.0%.

What it shows, with care: one capsule at a time, both VDMs caught every leaker and the VLM same backbone as d3 missed most of them, often at 0.9 or more confidence. What it does NOT show yet:
- **The block rows mix in pictures where a leaker is only partly in view** (a sliver at a crop's edge counts as "shown"); that is likely why 4 a picture reads worse than 64. Those pictures need their own label or to be left out before any block number means anything.
- **Decider 2B is miscalibrated on this task, not mis-wired** (corrected the same morning; the first reading said plumbing). Its request is built exactly as for d3 and Intern, which work. On the 7,500 leak_capsule questions it still ranks leakers above good capsules (leakers 0.9 to 1.0 on "leaker", good capsules mostly 0.8 to 0.9), but every answer leans to "leaker": it was confidently wrong (0.9 or more) on 6,113 good capsules. Method: the leaker probability binned by truth over `arms.jsonl`.
- Accuracy alone hides all of this: d3 97.4%, Qwen 93.2%, because most pictures are clean.

**Relabelled (Nov 5, late morning).** `python scripts/relabel_capsule_blocks.py output/capsules/sel01` (rule of e5ef177; original kept as `decisions.before_relabel.jsonl`) changed the 2,961 leak_block labels: 313 stay yes, **233 yes become None** (a leaker only partly in the picture; left out of scoring), 2,415 stay no. Recounted with `scripts/capsule_catch_rates.py`, the live run's questions now hold 595 leaker pictures:

| | d3 27B (VDM) | Intern 4B (VDM) | Qwen3.8-27B first word (VLM) | Qwen3.8-27B stated (VLM) |
|---|---|---|---|---|
| leakers caught (missed at 0.9 or more) | **92% (7)** | **98% (2)** | 43% (117) | 66% (202) |
| false alarms on 9,774 clean pictures | 0.7% | 6.0% | 2.0% | 3.1% |

The block-size rows above still use the old rule; the recut below replaces them. Still exploratory (selection data).

**Block curve recut with the e5ef177 rule (Nov 5, afternoon).** Method: each `blocks/n*/decisions.jsonl` copied to `decisions.before_recut.jsonl`, then `.\run.ps1 scripts/capsule_block_curve.py --log output/capsules/sel01 --max-scans 20` (log `sel01/curve_recut.log`). Checked: every row's fields other than `truth` are identical to the old cut; the only moves are yes -> None (4 a picture 72, 16 a picture 36, 64 a picture 13). Recounted from the existing replays with `python3 scripts/capsule_catch_rates.py` on each cut (log `sel01/catch_rates_recut.log`); no model was rerun.

| leakers caught (missed at 0.9 or more) | d3 27B (VDM) | Intern 4B (VDM) | Qwen3.8-27B first word (VLM) | Qwen3.8-27B stated (VLM) |
|---|---|---|---|---|
| one capsule a picture, 38 | **100% (0)** | 100% (0) | 26% (8) | 47% (20) |
| 4 a picture, 48 (72 left out) | 96% (1) | 96% (0) | 67% (5) | 88% (6) |
| 16 a picture, 41 (36 left out) | 80% (1) | 100% (0) | 66% (6) | 85% (6) |
| 64 a picture, 34 (13 left out) | 88% (1) | 97% (0) | 76% (4) | 94% (2) |
| whole tray, 17 | 82% (1) | 100% (0) | 76% (1) | 82% (3) |

False alarms on clean pictures, same sizes in order: d3 0 / 0.3 / 0.9 / 0 / 0%; Intern 0.7 / 4.7 / 7.6 / 8.0 / 0%; Qwen first word 0.2 / 0 / 0 / 0 / 0%; Qwen stated 0.4 / 0.8 / 1.2 / 0 / 0%. Decider 2B answers "leaker" on nearly everything (false alarms 100 / 90 / 63 / 49 / 100%).

What it shows: with the slivers left out, "4 a picture reads worse than 64" is gone (d3 96% at 4); the VDMs' misses are mostly unsure ones (d3: 1 confident miss per size at most); Qwen stated's misses are nearly all confident (20 of 20 on single capsules), and Qwen first word's are mostly unsure but it misses far more (28 of 38 single capsules, 8 of them confident). What it does NOT show: anything past 20 trays (17 to 48 leakers a size, so one picture moves a rate by 2 to 6 points), or the neighbour-leaker pictures below, which these cuts still label yes.

**The 61 neighbour-leaker pictures (Nov 5, afternoon; exploratory, no label changed).** Question (review round 7): a leak_block picture is the block's own capsules plus a 15 mm pad, and the rule says yes when any leaker lies wholly in it, so a leaker from the next block that sits inside the pad makes it yes. A leaker differs from a good capsule only by its pool (16, 22 or 30 mm across, centred on it), and a capsule wholly in the pad can still have its pool cut by the edge. Method: `python3 scripts/capsule_neighbour_leakers.py output/capsules/sel01` (pool box from `POOL_SIZES` and the logged `px_per_m`; log `sel01/neighbour_leakers.log`). Numbers: of the 313 yes pictures, 61 hold a neighbour's leaker wholly; 9 of those also hold one of their own (yes either way), **52 are yes only because of the neighbour**: in 42 the neighbour's pool is cut by the edge, in 10 it is whole. By leaker (43 in the 42 cut pictures, one holding two): pools cut 14 / 16 / 13 at sizes 16 / 22 / 30 mm, pools whole 9 / 1 / 0. All of these counts are printed by the command above (since the review fix; the first commit printed only the three kinds). How the arms answered them (said yes; own-leaker pictures for comparison):

| | own leaker, 261 | neighbour, pool cut, 42 | neighbour, pool whole, 10 |
|---|---|---|---|
| d3 27B | 95% | 57% (4 no at 0.9+) | 30% (3) |
| Intern 4B | 100% | 88% (1) | 80% (1) |
| Qwen first word | 71% | 43% (13) | 10% (6) |
| Qwen stated | 96% | 71% (12) | 50% (5) |

What it shows: every arm says yes far less on these than on its own-leaker pictures, so they read as unclear pictures with a yes label, and they hold 17% of the yes labels behind the live-run catch rates above. What it does NOT show: the block cuts (their rows do not log the pixel window; the cut would need the rule itself), or which rule dup picks; nothing is relabelled until he does.

## One station, no driving: reach, speed and the grasp (Nov 5, afternoon; probes, not results)

Asked by dup: "we need to cut all of the driving" (the robot drove twice per leaker, about 30 s of sim time each), "I don't really like how the sim doesn't run in real time", and "the leakers are floating away from the hand".

- **Reach with the hand turned** (`.\run.ps1 scripts/probe_reach.py --z 0.88 "--x=0.15,0.85,0.10" "--y=-0.70,0.70,0.10" --yaw-deg 45`, `output/bench/speed/reach_yaw45.json`): the wrist reaches 0.15 to 0.45 m ahead and 0.1 to 0.5 m to its side, as with the hand straight. The arm's length is the limit; tipping the hand or raising the torso was not run (it can only shorten the reach at table height).
- **One station** (`.\run.ps1 scripts/probe_station.py --every 3 --out output/cells/station`, log `output/cells/station.log`): parked once at cell a's origin (35 mm off), each arm on its side of the tray, hand headings 0 / 30 / 60 / 90 degrees toward the middle. **86 of 87 sampled capsule spots reached within 5 mm** (62 straight, 21 at 30, 3 at 60); the miss is spot 30 (near row, right). Every bin spot tried (x 0.25 to 0.40, |y| 0.30 to 0.36) was reached by the hand on its side. The Nov 4 "15 cm band" came from planning one hand, straight ahead only. Built in 72a8c9f (fix 5bad94e).
- **Speed, one shift, seed 1000, 136.9 s of sim** (the old driving layout; `output/bench/speed/`): demo config with the viewer 0.41x real time; viewer without the four extra robot cameras 0.28x; viewer with quarter meshes 0.25x; no viewer 0.83x. The viewer, not the cameras or the mesh detail, takes most of it, and even headless the physics runs below real time. One run each, so the 0.25 to 0.41 spread is within run-to-run noise as far as this shows.
- **Why the capsule floats** (`.\run.ps1 scripts/probe_pinch.py --out output/cells/pinch`): in the pick pose (palm toward the robot's middle, fingers forward, the sort task's side grasp) the fingertips sit 6 to 13 cm above the capsule's centre; the weld holds it under the palm's edge. Closing any finger in that pose leaves the tips 6 cm or more above it. Palm down (`--roll-deg 90`, `output/cells/pinch_roll90`), the index and thumb close downward and their tips meet about (0.115, -0.03, -0.05) m from the wrist. The fingertips are 11 to 17 mm wide and 14 to 26 mm thick (`output/bench/body.json`), and the tray leaves 5 mm between capsules across and 6 mm along: a pinch on the tray as packed puts the tips into the neighbours.

What it does NOT show: the station in a whole shift (run next), the right-hand reach of the near-right corner (spot 30), the palm-down reach over the whole tray, or any pinch closing on a capsule.

**The station in whole shifts (Nov 5, evening; headless, oracle, seeds 1000 to 1007).** Method: `.\run.ps1 scripts/run_capsules.py --episodes 8 --seed0 1000 --out output/capsules/station-headless` at the station with the travel-pose reset and lift retries below (log `output/capsules/station-headless.log`; failed picks now print and land in `checks.move_failures`). Numbers: 7 of 8 shifts ended success, 1 failure; 19 of 21 leakers in a bin. **Seed 1000's shift took 39.2 s of sim, against 136.9 s with the driving layout**; headless pace 0.66 to 1.01x real time. All three missed picks were the left hand on the tray's outer left column, far end (spots 254, 258, 259); two of those leakers ended off the tray and in no bin (the miss knocked them off), and **tray_clear still read yes**, because the check only looks at the tray.

Why (`scripts/probe_reach_stall.py`, logs `output/bench/reach_stall*.log`): no contact and no line fallback; the arm settles on the joints the IK chain found from the forearms-up pose, short of the target. From a ready pose over the middle of the hand's half (0.355, +-0.10), spots 254 and 69 reach within 1 to 3 mm, but **the far outer corners (258, 259 left; 9 right) stop about 31 to 35 mm short whatever the start, the hand heading (outward -30 and -60 tried) or the lift height (10 and 5 cm)**. The station probe's 86 of 87 overstated reach: it walked spot to spot, so each IK start was a neighbour's solution; a shift starts every pick fresh. Two fixes are in: the travel pose now zeroes every arm joint (the sort task's set only the elbows, and a twisted arm missed spot 69 twice), and the lift move repeats while it still closes in by more than 5 mm.

What it does NOT show: reach of every spot from a fresh start (only these corners were tested), or the viewer pace of the station.

**The fresh-start reach map, and why it drifted (Nov 5, night).** Method: `.\run.ps1 scripts/probe_station.py --fresh --every 2 --out output/cells/station_fresh` (each spot from the travel pose with the shift's own reach_over; log `output/cells/station_fresh.log`), then `--every 13` with the base pose logged per spot (`output/cells/station_drift.log`), then `scripts/probe_reach_stall.py --cell a --spot 0 --arm right` with the descent and contacts (`output/bench/stall0d.log`). Numbers: 52 of 130 spots reached, the far rows (x 0.40 and 0.46) missed everywhere, and by the end the right hand missed its own bin by 15 to 25 cm. **The robot had moved:** after the first spot the base was 17 cm back and turned 16 degrees, and it slid further on later reaches (30 to 40 mm on single far reaches). The lift-height reach moves nothing; **the descent to pick height at the near-right corner put the right wrist link against the tray's near and side walls and the right leaker bin's near wall**, and each descent pushed the base about 4 mm back and 0.3 degrees round. The sideways grip rides the wrist 12 cm behind the working point and only a few cm over the rim.

What it shows: the 52 of 130 is not the arm's reach (from a clean start, spot 69 reaches within 3 mm in both cells); contact between the wrist and the tray walls and bins shoves an unbraked base. What it does NOT show: how much the eight station shifts were shoved (their base pose was not logged), or whether the palm-down grasp clears the walls. An earlier `--fresh` run's 56 of 118 (`station_fresh_sweep.log`) also folded the arms from 4 mm over the tray; both runs are invalid as reach maps.

**Palm-down pinch, spaced tray, brake (Nov 5, late; headless, oracle, seeds 1000 to 1005).** Built: palm-down pinch welded to the index fingertip (`probe_pinch.py --roll-deg 90 --sweep`: open tips 50 mm apart, closed 39 mm on a capsule's ends, pinch point (0.126, -0.040, -0.052) m from the wrist), the tray spaced (0.065, 0.025) m for the fingertips (4 rows x 16, 64 capsules), bins at |y| 0.35, the base welded in place while parked, and a base-drift sensor in the run log. Method: `.\run.ps1 scripts/run_capsules.py --episodes 6 --seed0 1000 --out output/capsules/pinch-headless` (log `output/capsules/pinch-headless.log`). Numbers: **base drift 0.0 mm in all six**; 3 of 6 shifts success, 11 of 14 leakers in a bin. **Every failure was the left hand on its far rows** (spots 43, 46, 63: x 0.39 to 0.45, y 0.06 to 0.19), missing by 37 to 55 mm on both tries; the right hand reached its far row (spot 19). Shifts got slower: seed 1000 took 107.8 s of sim (three leakers) against 39.2 s with the sideways weld; headless pace 0.61 to 0.74x.

What it does NOT show: why the left hand misses palm down (not diagnosed), what costs the extra time (not measured), or the pinch in the viewer.

**Two half-trays inside the measured reach (Nov 5, night; headless, oracle).** Reach first: `.\run.ps1 scripts/probe_station.py --fresh --out output/cells/station_pinch` (palm down, straight ahead, each of the 64 spots of the spaced grid from the travel pose, base braked; log `output/cells/station_pinch.log`): **49 of 64 reached**; the base did not move. The misses were the middle band (within about 7 cm of the centre line on the near rows; the right hand gets slightly closer than the left) and the far outer corners. The grid is now two half-trays with a 15 cm aisle: 4 rows x 4 columns a side, 32 capsules, every spot inside that reach. A 16-capsule block is now a whole half, so no block picture shows another block's leaker (the 61-picture question no longer arises). Shifts: `.\run.ps1 scripts/run_capsules.py --episodes 6 --seed0 1000 --out output/capsules/halves-headless` (log `output/capsules/halves-headless.log`): **6 of 6 success, 0 failed picks, base drift 0 mm and 0 degrees**; seed 1000 (three leakers) took 59.6 s of sim; headless pace 0.70 to 0.84x. **13 of 14 leakers landed in a bin**: in seed 1001 the capsule struck the left bin's inner wall and fell on the table; leaker_landed answered no (correctly), but nothing acts on that answer and tray_clear only looks at the tray, so the shift still read success.

What it does NOT show: the viewer pace; why that drop hit the wall (drop error not logged); the full suite on this layout ran green (506 passed, 16 skipped).

## Where a capsule shift's time goes (Nov 5, night; one profile each, seed 1004, half-tray layout)

Asked by dup ("profile it and tell me the result", after asking why simulation videos run many robots in parallel while this one robot runs below real time). Method: `.\run.ps1 scripts/profile_shift.py --seed 1004 --config <config> --out output/bench/profile_<name>` (cProfile around one `run_capsules` shift; `shift.prof` and the log beside it). Figures are cumulative seconds: a call inside another counts in both.

- **Headless** (`config/handoff.yaml`): 114.3 s of sim in 173.9 s wall, **0.66x real time**, 15.2 ms wall per 10 ms step. IK solves 59.8 s (663 solves, 90 ms each, almost all waiting on the GPU result in the kit's `_numpy`); the physics step 91 s, of which **the robot's sensors (contact force, IMU with noise) 52.4 s, which the capsule task never reads**; camera pictures for the questions 1.8 s.
- **With the viewer** (`config/handoff_demo.yaml`): 100.6 s of sim in 465.3 s wall, **0.22x**, 46.2 ms per step. Viewer refresh 225 s (its render 118.6 s; **Genesis's on-screen help text alone 78.8 s**); visual update 148 s (moving the robot's five attached cameras, `move_to_attach`, 82.2 s; mesh poses 84.3 s); IK 120.3 s (565 solves, 213 ms each); sensors 79.8 s.

What it shows: none of the big costs is the capsules. The sensors and the robot's own cameras cost a third of the viewer run and the capsule task uses neither; the help text costs a sixth; each IK solve is a GPU round trip, and move_wrist solves one every centimetre of a move. What it does NOT show: the gain from turning each off (not run yet), run-to-run spread (one run each), or whether the CPU backend solves IK faster.

**After the cuts (Nov 6, just past midnight; same profile, seed 1004).** Capsule runs now build with no robot sensors, no robot cameras (`config/capsules.yaml`, `config/capsules_demo.yaml`), no viewer help text, and IK every 5 cm (`run_capsules.py` LEAN); the viewer draws at 30 Hz. Headless (IK every 3 cm, before the last two cuts): 100.6 s of sim in 75.8 s wall, **1.33x real time** (was 0.66x). Viewer, IK every 3 cm and 60 Hz: 0.51x. **Viewer, IK every 5 cm and 30 Hz: 100.7 s of sim in 164.8 s, 0.61x** (was 0.22x); left: viewer drawing 63.8 s, the question cameras' renders 35.3 s, IK 11.0 s. One run each; no picture the models see changed (the question cameras are scene cameras).

With the one-descent and pre-shape fixes (2917e62; `.\run.ps1 scripts/run_capsules.py --episodes 1 --seed0 1004 --config config/capsules_demo.yaml --out output/capsules/viewer-lean`, viewer, dup watching): 5 of 5 leakers in a bin, no failed move, no drift; **77.5 s of sim (was 100.7 s: the extra up-and-down on every pick) in 118.3 s wall, 0.66x**. Full suite at 2917e62: 509 passed, 16 skipped.

**Parked props moved aside; a rim bounce (Nov 6).** Unused props now wait 40 m aside as well as under the floor (dup saw them through the floor in the viewer). `.\run.ps1 scripts/run_capsules.py --episodes 3 --seed0 1003 --out output/capsules/park-check` (lean headless): 3 of 3 success, no failed move, no drift, **1.35 to 1.53x real time**. In seed 1003 one leaker caught the bin's rim on its first drop (leaker_landed no) and bounced back onto the tray; the rescan found it and the robot binned it, but the shift counted it twice (4 in the bin of 3), now counted once. What it does NOT show: how often a drop catches the rim (1 in 9 drops here).

## The selection log sel02 (Nov 6, 1:59 to 3:10 AM; oracle, lean headless, head level)

Method: `scripts/night_capsules.ps1` step 1, `run_capsules.py --episodes 150 --seed0 2000 --out output/capsules/sel02` at 46c72de (docs/FREEZE.md). Robot: 150 shifts in 71 min; base drift 0 mm in every shift; 2 failed moves. **sel02's lost leakers:** 278 of 299 leakers landed in a bin, **20 ended outside the bins** (130 shifts success, 20 failure; every failure is a lost leaker). Placed in each cell's frame (positions from `checks.leakers_loose`): 18 lay at |y| 0.18 to 0.30 m, between the tray's outer columns and the bin's inner wall (at 0.28), on the tray's edge or the table; 2 fell through the world (seeds 2067, 2149: metres under the floor, a fast drop tunnelling a collision surface). leaker_landed answered no for every one (truth from the simulator), so the log also holds real "no" landings. The hand reached the bin in all but 2 moves, so the loss is at the release: let go 2 cm above the rim, then the fingers opened, which can flick the free capsule off the near wall. Proposed (built, fake-tested, NOT run in Genesis, NOT in sel02, needs dup's call because it changes the robot between sel02 and test05): the pinch goes 3 cm below the rim inside the bin, the fingers open while the capsule is still welded, then it is let go.

**First replays on sel02 (Nov 6, 4:30 AM; EXPLORATORY: selection data).** Arms in: Intern-4B (3:11 to 3:27 AM) and d3 27B (4:01 to 4:23 AM, after the commit cap first skipped it), 0 errors for d3, 1 for Intern. Method: `python3 scripts/capsule_catch_rates.py output/capsules/sel02 output/capsules/sel02/blocks/{n1,n4,n16,nall}` (log `output/capsules/sel02_catch_rates_d3_intern.log`).

| leakers caught (missed at 0.9+) / false alarms | d3 27B | Intern 4B |
|---|---|---|
| live questions, 492 leaker pictures, 2,848 clean | 94% (1) / 25.1% | 71% (1) / 1.1% |
| one capsule a picture, 35 leakers, 605 clean | 97% (0) / 26.6% | 60% (0) / 0% |
| 4 a picture, 33 | 100% (0) / 0% | 100% (0) / 0% |
| 16 a picture (a half-tray), 23 | 91% (0) / 0% | 96% (0) / 5.9% |
| whole tray, 18 | 72% (0) / 0% | 94% (0) / 0% |

**Changed from sel01, and not yet explained:** on single capsules d3's false alarms rose from 0.7% to 26.6% and Intern's catch fell from 98-100% to 60%. d3's 161 single-capsule false alarms are all unsure (highest 0.76 on "leaker", below the sorting study's 0.80), so a ladder would hand them to a person rather than act; they cluster in the second and fourth columns of each half-tray (29 to 38 each against 3 to 13 in the others). Looked at by eye: a clean capsule with a faint grey ring of shadow round it, on bare parchment (the spaced tray leaves far more bare paper in a 45 mm crop than the packed one did). What it does NOT show: why the columns differ (lamp position, shadow, the nudges: not tested), whether a person would call those pictures clean at a glance, or anything past this one log.

**Shade goes with d3's single-capsule false alarms on sel02 (Nov 6, 4:45 AM; first titled "drives": a correlation, not a tested cause. Corrected Nov 7: with the shade gone in sel03, d3's single-capsule false alarms rose to 0.40 and the correlation is +0.30, see "The selection log sel03").** Method: `.\run.ps1 python scripts/capsule_shade_check.py output/capsules/sel02/blocks/n1` (median brightness of each crop's four corners, bare parchment, against d3's leaker probability). Over 605 clean single capsules the correlation is **-0.57**; crops with paper brightness 200 to 220 (of 255): **62% false alarms**, mean leaker probability 0.52 (n 149); brighter crops: 15%, 0.27 (n 456). Same row, next column: a lit capsule read 0.07, a shaded one 0.58 (`ep2003_s03_00105` vs `00109`). The whole-tray picture shows diagonal light and shade bands across the paper, most likely the robot's raised arms between the tray lamp and the tray. What it does NOT show: which body casts the shade (not traced), whether it also explains Intern's lower catch, or whether shadow-free lighting removes it (not run: the fix changes the pictures, so it means redoing sel02, dup's call).

**All four arms on sel02 (Nov 6, done 6:16 AM; EXPLORATORY: selection data).** Qwen first word 4:23 to 5:01 AM, Qwen stated 5:02 to 6:16 AM, every replay 0 errors (`output/capsules/replay_chain.log`). `python3 scripts/capsule_catch_rates.py` on sel02 and its cuts (log `output/capsules/sel02_catch_rates.log`). The live run's questions, 492 leaker pictures and 2,848 clean:

| | caught | missed at 0.9 or more | false alarms |
|---|---|---|---|
| d3 27B (VDM) | 94% | **1** | 25.1% (unsure; the shade, above) |
| Intern 4B (VDM) | 71% | **1** | 1.1% |
| Qwen3.8-27B stated confidence (VLM) | 71% | **145** | 3.5% |
| Qwen3.8-27B first word (VLM) | 55% | **62** | 4.2% |

One capsule a picture (35 leakers, 605 clean): d3 97% (0) / 26.6%, Intern 60% (0) / 0%, Qwen stated 63% (13) / 0.5%, Qwen first word 37% (5) / 0%. What it shows: both VDMs almost never miss a leaker while sure (1 each in 492); the VLM on d3's own backbone misses 62 to 145 while sure. What it does NOT show: the frozen ladder (not picked yet), anything on test05, or the same picture without the tray's shade bands.

Exploratory ladder pick on sel02 (`.\run.ps1 scripts/pick_ladder.py --log output/capsules/sel02 --human-cost 0.2`, defaults as PREREG.md; log `output/capsules/sel02_ladder_pick.log`, `sel02/ladder-pick.json`): **Intern-4B > d3 27B > person**, per question, mean value 0.9868 [0.984, 0.989], acted wrong 0.005, handed to a person 0.019. No Qwen arm enters the top ladders (adding Qwen stated as a third rung ties). Not frozen: the tray's shade question (above) comes first; a lighting fix means a new selection log.

**What casts the shade: the kit's directional sun (Nov 6, 6:40 AM; probe, not a change).** Method: `.\run.ps1 scripts/probe_tray_light.py --variant N --out output/cells/tray_light` for N = 0 to 3 (seed 2003's work cell, robot parked in the travel pose; pictures `v0.png` to `v3.png`, numbers `light.jsonl`). Bare-parchment brightness, median / 5th percentile / spread (95th minus 5th): **now** (sun 5, one lamp, ambient 0.3) 219 / 167 / 86; **no sun** 200 / 146 / 66; **no sun, 2 lamps a tray** 197 / 141 / 62; no sun, one lamp, ambient 0.6: 231 / 137 / 104. In `v0.png` long diagonal light and shade bands cross the tray from the robot's raised arm (its hand is in the frame's top right): the sun, low and to one side, throws the arm's shadow across the paper. With no sun and two lamps (`v2.png`) the bands are gone; the paper is evenly lit, a little dimmer, capsules and pools as clear; soft grey shadows remain at the halves' outer edges. A first run photographed the idle cell (empty tray) and an 8-lamp variant passed Genesis's light limit; both fixed. What it does NOT show: whether the models' false alarms fall without the sun (needs a new selection log), or a person's view of the dimmer paper.

## Drop fix and lighting, applied (Nov 6 morning; dup approved both)

- **Lighting** (06ae848): each tray lit by two lamps (one over each half, intensity 1.5 each, 1.2 m above the table), no sun, ambient 0.3. One shift filmed from two side cameras (`.\run.ps1 scripts/watch_shift.py --seed 1004 --every 1.5 --out output/capsules/watch`): 5 of 5 binned, a capsule seen on the bin floor (`frames/t0016.png`).
- **Drop fix** (c1e8cb5: pinch 3 cm below the rim inside the bin, fingers opened while welded, then released). `run_capsules.py --episodes 10 --seed0 2000 --out output/capsules/dropcheck`: 10 of 10 success, 13 of 13 binned (these seeds lost nothing in sel02 either). **The 10 seeds that lost a leaker in sel02** (2017, 2026, 2028, 2031, 2035, 2044, 2062, 2066, 2067, 2077; one-episode runs into `output/capsules/dropcheck2`): **10 of 10 success, 34 of 34 binned, 0 lost** (sel02: 10 shifts failed on them). Both with the new lighting.

What it does NOT show: the loss rate over 150 shifts (the redo will), or whether the lighting lowers d3's false alarms (needs the replays).

## Tile floor and steel table, capsule cells only (Nov 6 evening; a look change before sel03, dup's call)

dup, Nov 6: a whitish grey tile floor in place of the kit's blue checker and a steel table, "no shine if it would affect performance or results"; Phase 1 left as tested. Built: the kit's plane stays for physics but is not drawn in capsule scenes (`SceneSpec.kit_floor=False`, `simulator.py` `floor_visible`); a 24 m drawn-only slab carries 0.5 m tiles (`capsule_cell.floor_texture`); the table is steel grey (0.50, 0.52, 0.55) drawn matte (Rough), no metal surface, so the tray lamps leave no highlight on it.

Method: `.\run.ps1 scripts/probe_cell_look.py --look old` then `--look new` (seed 2003, robot parked in the travel pose, every scene camera plus a wide shot; pictures in `output/cells/look/{old,new}/`), then `--compare`. Parchment-only check: pixels [300:980, 450:1480] of each tray picture, old against new.

- **The parchment the leak questions crop from is unchanged:** max pixel difference 1 to 2 of 255 across two renders, 0% of pixels over 3 (`--compare`, `parchment_max_diff`; first measured by hand on the same crop, mean brightness 210.2 / 188.4 both ways).
- **What changes:** the table round the tray (tray pictures 39 to 42% of pixels), the bin cameras (51 to 55%: the table under and round each bin), the overview cameras (76 to 82%: floor and table). So where_needed and leaker_landed pictures change, and leak_block crops that reach past the tray rim show grey table where brown was. sel03 is made with this look, so its pictures and test05's match.
- The whole-picture "paper" figures (p05 150 to 126) count grey table pixels as paper: their pixel count grows 970k to 1.28M (a_tray). The parchment-only check above is the real one.
- **Speed:** scene build 46.2 s old / 44.1 s new, captures the same (overview 1.55 / 1.52 s first frame, others 0.02 to 0.05 s). One more drawn box; no measurable cost.
- Seen in the pictures: soft grey shadows of the robot on the parchment under the two lamps, the same in old and new (pixel-identical), not from this change.

Re-render with the plain ground (043f242): it runs on past the tiles to the horizon with no flicker in `look_wide.png`; build 47.2 s.

What it does NOT show: viewer frame rate with the new floor (dup's viewer look, next), or anything about the models' answers on the new look (sel03's replays will).

## Qwen3.8-27B on the finish question, test03 (Nov 6 night; EXPLORATORY: selection data, for the film's Phase 1 comparison)

Why: the film explains how a decision model answers by putting a VLM and a VDM on the same picture; dup wanted a Phase 1 painted part there instead of a capsule. Qwen3.8-27B (d3's base model) had never been replayed on the sort logs (test03 held only d3's family, test04 only d3-27b-vision).

Method: `python3 scripts/sublog.py output/sort/test03 output/sort/test03_finish --question finish` (the 270 finish decisions, their frames, and the arm rows already replayed on them, renumbered; test03 itself untouched), then `.\run.ps1 scripts/replay_capsules.py --logs C:\...\output\sort\test03_finish --arms 'qwen-27b-vision,qwen-27b-self-vision'` (270 answers each, 0 errors; 141 s and 261 s). Counted from `test03_finish/arms.jsonl` against each decision's truth (good / defective, folded).

| 270 finish pictures | right | wrong while 0.9+ sure |
|---|---|---|
| d3 27B vision (VDM) | 93.3% | 1 |
| Qwen3.8-27B, first word (VLM) | 94.1% | 6 |
| Qwen3.8-27B, stated confidence (VLM) | 95.2% | 12 |

What it shows: on the finish question the VLM is as accurate as d3, but six to twelve times as often sure and wrong, the case a confidence threshold cannot catch. The film's picture is decision 169 (`frames/ep0456_606_finish.png`, truth good): Qwen "defective" 0.954 (first word) and 0.95 (stated), d3 good 0.941. Picked to show the difference; 2 of 270 pictures had both Qwen readings wrong at 0.9+ with d3 right and sure (169 and 186). What it does NOT show: anything on test04 or a frozen protocol (selection data), or other questions.

## The selection log sel03 (Nov 6 to 7, 11:15 PM to 2:49 AM ET; oracle, lean headless; two lamps, drop fix, tile floor and steel table)

Why: sel02's pictures predate the lighting and drop fixes and the new floor, so test05 would not match them. sel03 is the selection log the capsule PREREG is filled from (docs/FREEZE.md, same seeds 2000 to 2149).

Method: `powershell -File scripts\night_capsules.ps1` at 2a7f1ff (clean tree; 91f64d0 plus a film-only commit): `run_capsules.py --episodes 150 --seed0 2000 --out output/capsules/sel03`, `capsule_block_curve.py --sizes "1,4,16,0" --max-scans 20`, `replay_capsules.py` with the four arms (one model on the GPU at a time), `capsule_catch_rates.py`. Logs `output/capsules/night.log` (UTF-16), `sel03.log`, `sel03_replay.log`, `sel03_catch_rates.log`. Windows commit 72.1 GB at launch; the native Ollama's embeddinggemma unloaded first.

**Robot (150 shifts in 62 min of shift time, 65 min wall):** 145 success, 5 failure. 299 leakers, **294 binned**, 4 lost outside the bins (seeds 2011, 2016, 2060, 2121), 1 left on the tray after two missed picks (seed 2068, left hand, "missed the capsule by inf mm"). sel02 had 20 lost on the same seeds: the drop fix holds over 150 shifts. 0 good capsules thrown out, base drift 0 mm in every shift. Where the 4 lost ones lay was not traced.

**Replays:** 3,751 answers per arm on sel03 plus 640 / 160 / 40 / 20 on the cuts n1 / n4 / n16 / whole tray, **0 errors in every replay**. d3 18 min, Intern 11 min, Qwen first word 31 min, Qwen stated 61 min on sel03 itself.

**Catch rates on sel03's live questions** (486 leaker pictures, 2,816 clean; EXPLORATORY, selection data):

- d3-27b-vision: caught 0.87, confidently missed (>= 0.9) 1, false alarms 0.337 (sel02: 0.94, 1, 0.251)
- intern-4b-vision: caught 0.84, confidently missed 0, false alarms 0.024 (sel02: 0.71, 1, 0.011)
- qwen-27b-self-vision: caught 0.65, confidently missed 169, false alarms 0.036 (sel02: 0.71, 145, 0.035)
- qwen-27b-vision: caught 0.45, confidently missed 95, false alarms 0.043 (sel02: 0.55, 62, 0.042)

On the single-capsule cut (n1: 35 leakers, 605 clean) d3 catches 0.89 with 0.400 false alarms; Intern 0.74 with 0.000. On n4 and n16, Intern catches every leaker (33/33, 23/23) with no false alarms. The Qwen VLM arms miss leakers at 0.9 or more on every cut but the whole tray: **the VLM is sure and wrong 95 to 169 times on the live questions, the two VDMs 0 to 1 times**.

**d3's single-capsule false alarms are no longer shade-driven.** `.\run.ps1 python scripts/capsule_shade_check.py output/capsules/sel03/blocks/n1` (log `sel03_shade_check.log`): correlation of paper brightness with d3's leaker probability is **+0.30** (sel02: -0.57). Brightness 200 to 220: false alarm 0.44 (n 486); 180 to 200: 0.25 (n 109); 150 to 180: 0.00 (n 10). The two-lamp lighting removed the dark bands, and d3 now false-alarms more on the brighter crops. What it does NOT show: why d3 calls a third of clean single capsules leakers (not traced); the ladder hands those off by threshold either way.

**Ladder pick** (`.\run.ps1 scripts/pick_ladder.py --log output/capsules/sel03 --human-cost 0.2`, PREREG.md defaults; log `sel03_ladder_pick.log`, `sel03/ladder-pick.json`): **Intern-4B > d3 27B > person, per question**, mean judgement value 0.9836 [split range 0.981 to 0.988], acted wrong 0.005, confidently wrong 0.0005, handed to a person 0.029, alone and right 0.965. The same ladder as sel02's exploratory pick (fit per question here, "per question, under 50 pooled" there; that fit scores 0.9830 on sel03, inside the tie margin). Adding either Qwen arm as a third rung scores the same 0.9836 (more rungs, so not picked). Best single arm: Intern alone, 0.9732. Thresholds fitted on all 3,751 sel03 decisions (`freeze_ladder.py freeze --log output/capsules/sel03`, log `sel03_freeze.log`, `sel03/frozen-ladder.json`).

What it does NOT show: anything on unseen shifts (test05 will), real cameras or capsules, or the ladder's own robot path (the motions follow the oracle).

## test05: the capsule confirmation (Nov 7, 1:07 to 4:34 PM ET; scored once)

The protocol is docs/PREREG_CAPSULES.md, committed (0981889) before test05 existed. test05 was made at f36df07: `scripts/night_test05.ps1` ran `run_capsules.py --episodes 150 --seed0 3000 --out output/capsules/test05`, `replay_capsules.py` with the four FREEZE arms, then `freeze_ladder.py confirm --log output/capsules/test05 --frozen output/capsules/sel03/frozen-ladder.json`. confirm ran at 0e8f7b6, whose only change refuses a rung under 95% coverage; it passed with both rungs at 3785 of 3785. Nobody looked at test05's answers before confirm ran.

Logs are `output/capsules/night_test05.log`, `test05.log`, `test05_replay.log`, `test05_confirm.log` and `test05/confirmation.json`. confirm's printout still says "test04 [95%]" and "test03 prediction"; the values are test05's and sel03's. Qwen first word timed out on 2 answers (WinError 10060, decisions 417 and 909). It is not a ladder rung; those 2 were retried at 4:35 PM (`test05_replay_retry.log`) and every arm now has 0 errors.

**Robot (150 shifts, 67 min):** 145 succeeded. Of 281 leakers, 276 landed in a bin; 3 were lost outside the bins and 2 were left on the tray. 0 good capsules thrown out.

**The frozen ladder (Intern-4B > d3 27B > person, per question), 3,785 decisions, 95% by-shift intervals, against the sel03 prediction:**

- **Acted wrong: 0.002 [0.000 0.003]** (6 decisions), predicted 0.005
- Confidently wrong: 0.001 [0.000 0.002], predicted 0.0005
- Handed to a person: 0.028 [0.023 0.032] (104 decisions), predicted 0.029
- **Decided alone and right: 0.971 [0.967 0.976]**, predicted 0.965
- Answered by: Intern 3,398, d3 283, person 104. ECE 0.073 (ECE is how far stated confidence sits from how often the answer is right; 0 is perfect).

Every rate sits inside or better than its prediction. **The ladder's six wrong actions:**

- **2 leaker capsules passed as good**, both by Intern above its 0.80 threshold. That is 2 of the 467 leaker pictures; 389 were caught by a model and 76 went to a person.
- 2 clean blocks called leaky.
- 2 tray_clear calls: one "clear" with a leaker still on the tray, one "not clear" on a clear tray.
- By question: leak_block 0.993 right (286 decisions), leak_capsule 0.9993 right (2,926; 2.6% to a person), leaker_landed 1.000 (280), tray_clear 0.986 (143; 19.6% to a person), where_needed 1.000 (150).

**Every arm alone on the leaker calls** (leak_block and leak_capsule: 467 leaker pictures, 2,745 clean; `python3 scripts/capsule_arm_report.py output/capsules/test05 --json output/capsules/test05/arm-report.json`, log `test05_arm_report.log`; 95% by-shift bootstrap, 2,000 resamples):

| Arm | Leakers caught | Confidently missed (0.9+) | False alarms | All questions: wrong at 0.9+ |
|---|---|---|---|---|
| d3 27B (VDM) | 0.850 [0.815 0.884] | **1** (0.002 [0.000 0.007]) | 0.362 [0.345 0.378] | 1 of 3,785 |
| Intern-4B (VDM) | 0.850 [0.818 0.882] | **1** (0.002 [0.000 0.007]) | 0.001 [0.000 0.002] | 2 |
| Qwen3.8-27B, stated (VLM) | 0.634 [0.590 0.677] | **171** (0.366 [0.322 0.410]) | 0.000 [0.000 0.001] | 358 |
| Qwen3.8-27B, first word (VLM) | 0.422 [0.376 0.471] | **98** (0.210 [0.166 0.254]) | 0.000 [0.000 0.000] | 138 |

**The VLM on the same backbone as d3 was sure and wrong about 98 to 171 leakers; the VDMs, 1 each.** That matches sel02 and sel03.

`capsule_catch_rates.py` (the quick look, log `test05_catch_rates.log`) also counts tray_clear as a leaker call, which is why its numbers differ: 472 leaker pictures and 2,883 clean, and its false alarms are 0.020 for Intern and about 0.04 for Qwen because of tray_clear.

What it does NOT show:
- Real cameras, capsules or a robot. A pool is the only sign of a leaker; there are no look-alikes and one lighting.
- What happens to a leaker lost outside a bin: the shift fails, and nobody is handed it.
- The ladder's own robot path: the motions follow the oracle, so the ladder's answers were scored, not acted on.
- Why d3 calls 36% of clean pictures leakers. That is untraced; the ladder sends those to a person or to Intern first.
