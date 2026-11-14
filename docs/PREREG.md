# Confirmation protocol (written before test04 exists)

This file is committed before the confirmation log is generated, and its commit hash goes in every confirmation result. Nothing here changes after test04 is made; a change means a new log (test05).

## What is frozen before test04 is opened

- The ladder: which arms, in which order. Chosen on test03 held-out episodes only, by `scripts/pick_ladder.py` (filled in Oct 19, see Frozen values below).
- The threshold fit and its settings: fit mode, `min_decisions` and held-out fraction (Frozen values below). The cost is set: a wrong action costs 10 times a hand-off to the person (dup, Oct 16, from seat final inspection: a bad seat that ships means an angry email from Chrysler to management and the whole line told). In the objective (right +1, wrong -1, person 1 - `human_cost`) a wrong action costs 2 and a hand-off costs `human_cost`, so 10 times means **`human_cost` 0.2**.
- The headline is quoted on the judgement questions (all but `safe_to_grasp`, which a hard rule decides before any model is asked).
- The thresholds themselves: fitted on all of test03, written into this file as numbers (Frozen values below).
- The prompts, state text, model files and server versions (tags and digests).

## The confirmation log

- `test04`: 300 episodes, seeds 1000 to 1299 (no earlier log used a seed above 489), fault cycle (100 clean, 100 forced drop, 100 crowded), same scene and settings as test03 (`run_sort.py --episodes 300 --seed0 1000 --fault cycle --ladder oracle --out output/sort/test04`).
- Size: 3,262 labelled decisions in 300 episodes (counted Oct 19 from `decisions.jsonl`, before any arm answered them); the judgement share is about 0.69 on test03, so about 2,250 judgement decisions. With the design effect measured on test03 (about 1.6 from decisions sharing an episode), that puts a 95% interval of about plus or minus 1 point on the acted-wrong rate (3 to 5% on test03) and about plus or minus 0.5 point on confidently wrong (about 1%). 90 episodes would give about plus or minus 2 points and 1 point.
- Only the frozen ladder's arms are replayed on it. Nobody looks at test04 answers before the protocol above is filled in and committed.

## Frozen values (filled in Oct 19, 2026, before any arm was replayed on test04)

**Ladder: `d3-27b-vision` > person.** One rung. The machine acts when d3 27B's top probability is at least the threshold; otherwise the person decides.

- How it was chosen: `scripts/pick_ladder.py --log output/sort/test03 --human-cost 0.2` (defaults: 10 splits, held-out fraction 0.3, seeds 0 to 9, `min_decisions` 50, rung pool = the 10 best arms alone, up to 3 rungs). It scores every cheap-to-expensive chain of the pool under three threshold fits on each split's test episodes, with thresholds fitted on that split's held-out episodes. Rule: highest mean judgement value (right alone +1, wrong alone -1, person 1 - `human_cost`); within 0.005 of it, fewest rungs, then the cheapest first rung. Best: `d3-27b-vision` alone, per-rung fit, 0.9230 (split range 0.899 to 0.947). Nearest multi-rung ladders: d3 27B > StartLux 27B 0.9220, d3 27B > OneJev 27B 0.9215 (within the tie margin, more rungs). Full list: `output/sort/test03/ladder-pick.json`.
- Fit: **per rung** (one threshold for the rung, all questions), threshold grid 0.30 to 1.00 in steps of 0.05, `human_cost` **0.2**. (`min_decisions` and held-out fraction apply only to the pick, above; the frozen fit uses all of test03.)
- **Threshold: `d3-27b-vision` 0.80**, fitted on all 974 labelled test03 decisions (`scripts/freeze_ladder.py freeze`, written to `output/sort/test03/frozen-ladder.json`).
- Model: `d3-27B-Artemis-IQ4_XS.gguf` (turintech/d3-27B-Artemis-GGUF, 14,949,286,144 bytes, sha256 `4ae3fc00c3f7e7475796263a2b2c6311d2090f07f1dbcae9a175faa215366568`) with `mmproj-vllm-d3-BF16.gguf` (ggml-org/vllm-d3-GGUF, 931,146,240 bytes, sha256 `e196a47584d33044d1f9b4aefd25326e35ec1388118da31160e383c0f46f13d2`). Decision temperature as stored in the GGUF: choice 1.0 (`qwen35.decision.type` vllm-d3), not overridden.
- Server: llama.cpp release b11723 (`build_info` b11723-31288c9c6, Windows CUDA build, `llama-server.exe` sha256 `86171e23d70a33d7603584826c90906e45eaeac1a53b48b705f5629e215758e6`), `-c 16384 -ub 8192 -b 8192 -np 1 -ngl 999`, `/v1/systemone`, the frame as a data-URL image (`D:\Toolkits\build\serve_d3-27b.cmd`), one RTX 3090 on native Windows.
- Prompts and state: the question texts and option criteria are `QUESTIONS` in `src/handoff/decider.py`; the request body is `DeciderRung` in `src/handoff/rungs.py`; the state text and frame of every decision are as recorded in `output/sort/test04/decisions.jsonl` when the log was made. All at the commit that adds this section.
- Prediction (test03, mean of the 10 splits' test episodes): overall acted wrong 0.018, confidently wrong 0.010, handed to the person 0.311, decided alone and right 0.671; judgement acted wrong 0.018, confidently wrong 0.014, person 0.207, alone and right 0.775.
  - *Note added Oct 19-20, after the confirmation (the frozen values above are unchanged):* this prediction is the picked candidate's own ten-split score, the best of 525 ladder × fit candidates on those same splits, so it could be optimistic (winner's curse). A nested split of test03 (`scripts/nested_pick.py`, VALIDATION.md) measured no such bias: judgement acted wrong predicted minus observed +0.006 (sd 0.018).

**Procedure:** replay only `d3-27b-vision` on test04 (`benchmark.py replay --log output/sort/test04 --arm d3-27b-vision`, all 3,262 decisions), then `scripts/freeze_ladder.py confirm --log output/sort/test04 --frozen output/sort/test03/frozen-ladder.json` scores it with the threshold above, no refit, and reports the four rates with 95% by-episode intervals beside the prediction. Run once; errored calls are retried, nothing else is rerun.

## What gets reported

Each with a 95% by-episode bootstrap interval, overall and on the judgement questions (all but `safe_to_grasp`, which a hard rule decides):

1. Acted wrong: answered without the person, and wrong. The number a factory cares about.
2. Confidently wrong: acted wrong at 0.9 confidence or more.
3. Handed to a person.
4. Decided alone and right.

Beside each, the test03 prediction (ten-split mean), so a reader sees whether the replay estimate held.

## What it still does not show

Live behaviour (every decision sits on the oracle's path; the live-ladder check covers that), real cameras, real parts. Simulation only.
