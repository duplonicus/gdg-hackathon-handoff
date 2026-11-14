# Capsule cell: confirmation protocol (written before test05 exists)

**Status: filled from the selection log sel03** (Nov 7) and committed before test05 is generated. sel02 was the planned selection log; its pictures predate the two-lamp lighting, the drop fix and the tile floor, so sel03 (same seeds 2000 to 2149, same commands) replaced it (VALIDATION, "The selection log sel03"). This file's commit hash goes in every test05 result. Nothing here changes after test05 is made; a change means a new log (test06).

## The study

The capsule cell asks one question of every model: does this picture show a leaking soft-gel capsule? It compares two kinds of model on the same pictures:

- **VDMs (vision decision models):** read the picture and return a probability for each option. d3-27b-vision (built on Qwen3.8-27B) and Intern-Decision-4B.
- **A VLM (vision-language model) on the same backbone as d3:** Qwen3.8-27B, read two ways: the probability of its first answer word, and the confidence it states when asked.

The robot acts alone when the model is sure enough; otherwise a person decides. The question is which kind of model a plant can trust to act alone.

## What is frozen before test05 is opened

- The scene, the robot's moves, the cameras, the questions and the block size: `docs/FREEZE.md` at the commit that adds this file.
- Which arm leads the ladder and its threshold: chosen on sel03 only, by the same rule as the sorting study (highest mean value with a wrong action costing 10 times a hand-off, `human_cost` 0.2; PREREG.md), held-out splits by shift.
- The headline questions: leak_block and leak_capsule (the leaker calls) and leaker_landed (did the drop land). where_needed and tray_clear are reported beside them.
- The model files, server builds and request bodies, as listed under Frozen values.

## The confirmation log

- **test05:** `run_capsules.py --episodes 150 --seed0 3000 --out output/capsules/test05`, oracle answers, lean headless, head level. Seeds 3000 to 3149; no earlier log used a seed above 2149.
- Only the four arms of `docs/FREEZE.md` section 4 are replayed on it (the ladder's two plus the two Qwen readings, for the side-by-side). Nobody looks at test05 answers before this file is filled in and committed.

## Frozen values (from sel03)

- **Ladder: intern-4b-vision > d3-27b-vision > person, thresholds per question.** A rung acts alone when its top answer's probability is at or above its threshold for that question; otherwise the next rung is asked, and after d3 a person decides. Fitted on all 3,751 sel03 decisions (`output/capsules/sel03/frozen-ladder.json`):

| Question | Intern-4B | d3 27B |
|---|---|---|
| leak_block | 0.30 | 1.00 |
| leak_capsule | 0.80 | 0.85 |
| leaker_landed | 0.90 | 0.30 |
| tray_clear | 0.80 | 0.65 |
| where_needed | 0.50 | 0.30 |

- **How it was chosen:** `.\run.ps1 scripts/pick_ladder.py --log output/capsules/sel03 --human-cost 0.2` at 2a7f1ff (defaults as PREREG.md: 10 splits by shift, held-out fraction 0.3, seeds 0 to 9, `min_decisions` 50, up to 3 rungs; 42 ladder x fit candidates). Best 0.9836; adding qwen-27b-vision or qwen-27b-self-vision as a third rung ties at 0.9836 (more rungs, not picked); the same two rungs with the "per question, under 50 pooled" fit 0.9830 (margin 0.0006). Best single arm: Intern alone 0.9732. Thresholds: `scripts/freeze_ladder.py freeze --log output/capsules/sel03`.
- **Models** (sha256 read Nov 7 with `Get-FileHash`):
  - d3 27B: `d3-27B-Artemis-IQ4_XS.gguf` (14,949,286,144 bytes, `4ae3fc00c3f7e7475796263a2b2c6311d2090f07f1dbcae9a175faa215366568`) with `mmproj-vllm-d3-BF16.gguf` (931,146,240 bytes, `e196a47584d33044d1f9b4aefd25326e35ec1388118da31160e383c0f46f13d2`); the same files as PREREG.md.
  - Qwen3.8-27B: `Qwen3.8-27B-UD-IQ4_XS.gguf` (unsloth, 14,252,845,984 bytes, `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`) with `mmproj-BF16.gguf` (931,146,432 bytes, `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`).
  - Intern-Decision-4B (internlm, on Qwen3.5-4B), `D:\Models\decision-servers\intern\model`: `model-language-0001` (4,259,095,264 bytes, `ea71a535feda64b433d048d8d773637d9b1e5f317b832af531c70221159770ac`), `model-language-0002` (4,152,470,040, `c0a64383b5e5ca18c1086135dba5ee123a6c203ad86ffa4a08bdb9107ef61d6e`), `model-projector-0001` (54,543,992, `a918679313b12c1d31659cd3b492e8be53c616217c2f7ec658ed7f0fdfa22ee0`), `model-vision-0001` (612,517,440, `cf1793360eb8e8f1de0678df1a4c8b0b9042ae79e618aff1fa24c30f3b41b233`), all `-save_rank0.safetensors`.
- **Servers:** both 27B models on llama.cpp b11723 (`llama-server.exe` sha256 `86171e23d70a33d7603584826c90906e45eaeac1a53b48b705f5629e215758e6`, as PREREG.md), `-c 16384 -ub 8192 -b 8192 -np 1 -ngl 999` (`D:\Toolkits\build\serve_d3-27b.cmd`, `serve_qwen-27b.cmd`). Intern by `scripts/intern_server.py` in its own venv (torch 2.9.1, transformers 5.14.1, the Conv3d-as-matmul fix; VALIDATION Oct 18). Request bodies: `scripts/replay_capsules.py` at the commit that adds this file. One RTX 3090, native Windows.
- **Prediction** (sel03, mean of the 10 held-out splits): judgement value 0.9836 (split range 0.981 to 0.988), acted wrong 0.005, confidently wrong 0.0005, handed to a person 0.029, decided alone and right 0.965.

## What gets reported

Each with a 95% by-shift bootstrap interval, on the leaker calls and on all questions:

1. **Leakers caught**: the share of leaker pictures answered "leaker" or "yes".
2. **Confidently missed**: leaker pictures answered "good" or "no" at 0.9 or more. The number that ships a leaker.
3. **False alarms**: clean pictures called leakers.
4. With the frozen ladder: acted wrong, handed to a person, decided alone and right.

The same four for every arm replayed, so VDM and VLM sit side by side on the same pictures.

## What it does not show

Real cameras, real capsules or a real robot; a pool is the only sign of a leaker here (no look-alikes, matte parchment, one lighting). Leakers dropped outside a bin fail the shift but are not handed to a person. The robot's motions are on the oracle's path (every answer right), not the ladder's.
