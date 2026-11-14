# Paper notes

Working file for the write-up of our findings. Results go in here the day they are measured, with the log they came from. Nothing here is public until the hackathon is over.

## Title (not chosen)

- **Who Answers? Confidence-Gated Handoffs Between a Small Decision Model, a Vision-Language Model and a Human Operator on a Humanoid Sorting Task** (Claude's pick: it names the finding, which is about who should answer, not only how good one model is)
- Evaluating the Efficacy of Decision Models in Humanoid Robots (dup's first draft)
- Small, Calibrated, and Willing to Ask: Decision Models as the Handoff Gate for a Humanoid Robot

## The claim, in one paragraph

A robot working beside people should not act on a judgement it is unsure of. We put a ladder between the robot's fixed logic and its judgement calls: a small open decision model answers first, a larger model second, a person last, and each rung answers only when its confidence clears a threshold measured for that rung and that question. The useful numbers are not accuracy but the pair: how often the robot is confidently wrong, and how often it has to ask a person. On 974 recorded decisions, a ladder of two open decision models that read the camera frame (a 2B and a 27B) decides 79 percent of decisions alone and right, hands 13 percent to a person and is confidently wrong on 0.6 percent, with no vision-language model in the loop; the ladder that uses Gemma as its second rung ties it on the first two numbers and is confidently wrong more often. The small decision model alone is never confidently wrong on 676 scored decisions.

## What is new here

- Decision models applied to robot handoffs, with images. The closest prior work (REFLEX, arXiv 2609.26532) uses a decision model with one fallback for text agents; nothing on robots, vision, or a person as a rung.
- A ladder of any length with thresholds measured per rung and per question, and each rung scored on the questions escalated to it.
- The metric pair: confidently-wrong rate and share handed to a person.
- Labels from the simulator's ground truth, so every decision is scored without human annotation.
- (Planned) fine-tuning the small decision model on its own logged decisions, an on-device compute budget, cloud models as a comparison, and learning from failures.

## Setup

- Robot: Mirsee MH3 (wheeled humanoid) in the Genesis simulator, 1.4.3.
- Task: sort parts by paint finish into a good bin and a rework bin. The question is good or defective (since Oct 12; it was good / scratched / unpainted, and a person could not tell the two defects apart in the frames; which defect a part has stays in the log). Parts arrive from a stop-and-go pick station. Forced failures: a part dropped mid-carry, a crowded station.
- Control: a behaviour tree with scripted motion. Four questions go to the ladder: finish, safe_to_grasp, in_bin, sort_complete. Each has a fixed option list, a hard rule where safety is involved, and a safe option used when nobody answers.
- Cameras: a fixed inspection camera over the pick station (finish), a fixed camera over the bins (in_bin, sort_complete). The robot's head camera is not used for any question yet.
- Bins: moved up to 2 cm each way per episode. The robot finds them from the bins camera (tray colour at wall-top height in the depth point cloud) and aims at what it saw; ground truth uses where they really are. Since Oct 12: located within 1.0 mm in 40 of 40 bins (mean 0.5 mm, bins moved 14.5 mm on average, simulated depth with no noise); 19 of 20 clean sorts, the one failure a part dropped off the table mid-carry.
- Models: Strands Decider 2B with vision, and its text-only E2B and E4B siblings; Perplexity pplx-decider v1.1 27B (TurinTech Artemis 4.5-bit GGUF, with the frame); Gemma 4 e2b, e4b and 12b through Ollama, by first-token probabilities and, for e2b and e4b, by self-stated confidence. All local, on one RTX 3090 (cloud arms still to run).
- Method: decisions are recorded once with their frames and state text, then replayed through every model.

## Results (test03, Oct 13 to 14)

Test set `output/sort/test03`: 90 episodes with the bins moved per episode and located by the robot, 974 decisions (safe_to_grasp 300, in_bin 280, finish 270, sort_complete 124), recorded with the oracle answering so every decision has its truth. Every arm replayed all 974. Thresholds are chosen per rung and per question on 27 held-out episodes and the rest, 676 decisions, is scored. "Confidently wrong" means answered at 0.9 confidence or more and wrong. Intervals are 95 percent by-episode bootstrap. The 300 safe_to_grasp decisions are decided by a hard rule in the robot; every arm sits at the base rate there, so the "judgement" figures leave them out. Full tables with the per-question rows: `VALIDATION.md`; raw: `output/sort/test03/benchmark.json`.

| Arm (676 scored) | Right [95%] | Confidently wrong [95%] | Calibration error |
|---|---|---|---|
| pplx-decider 27B, frame | 0.805 [0.76, 0.85] | 0.016 [0.00, 0.03] | 0.068 |
| Decider 2B, frame | 0.780 [0.73, 0.83] | 0.000 [0.00, 0.00] | 0.078 |
<!-- STALE (Oct 16): this headline is one split (seed 0) and leaves out the acted-wrong rate. Ten-split figures and the person-cost curve: VALIDATION.md, "Headline ladder over ten splits". Rescore after the evals and the test04 confirmation. -->
| Gemma e4b | 0.768 [0.75, 0.79] | 0.195 [0.17, 0.22] | 0.191 |
| Gemma e4b, own confidence | 0.720 [0.70, 0.75] | 0.280 [0.25, 0.30] | 0.264 |
| Decider E4B, text + description | 0.562 [0.53, 0.60] | 0.034 [0.02, 0.05] | 0.253 |
| Decider 2B, text + description | 0.547 [0.52, 0.58] | 0.003 [0.00, 0.01] | 0.262 |
| Decider E2B, text + description | 0.543 [0.51, 0.57] | 0.028 [0.02, 0.04] | 0.324 |
| Gemma e2b, own confidence | 0.470 [0.45, 0.49] | 0.530 [0.51, 0.55] | 0.478 |
| Gemma e2b | 0.464 [0.44, 0.49] | 0.522 [0.49, 0.55] | 0.528 |
| Gemma 12b | 0.359 [0.31, 0.41] | 0.407 [0.36, 0.45] | 0.533 |

Ladders in three families. A: decision models reading the frame, no vision-language model anywhere. B: decision models reading the state text plus Gemma e4b's one-sentence description of the frame. C: Gemma itself as a rung. "Decided alone and right" is the share of all scored decisions the models answered without the person and got right; the person is the simulator's truth.

| Ladder | Decided alone and right [95%] | Handed to a person [95%] | Confidently wrong [95%] |
|---|---|---|---|
| A: Decider 2B, then pplx-decider 27B, then a person | 0.793 [0.75, 0.83] | 0.13 [0.11, 0.14] | 0.006 [0.00, 0.01] |
| C: Decider 2B, then Gemma e4b, then a person | 0.786 [0.75, 0.82] | 0.13 [0.11, 0.15] | 0.009 [0.00, 0.02] |
| C: pplx-decider 27B, then Gemma e4b, then a person | 0.672 [0.64, 0.70] | 0.28 [0.26, 0.31] | 0.010 [0.00, 0.02] |
| A: pplx-decider 27B, then a person | 0.645 [0.62, 0.67] | 0.32 [0.29, 0.34] | 0.007 [0.00, 0.02] |
| B: Decider 2B frame, then Decider E4B text, then a person | 0.620 [0.59, 0.65] | 0.28 [0.26, 0.31] | 0.001 [0.00, 0.00] |
| A: Decider 2B, then a person | 0.571 [0.54, 0.60] | 0.36 [0.34, 0.38] | 0.000 [0.00, 0.00] |
| C: Gemma e4b, then a person | 0.521 [0.49, 0.55] | 0.41 [0.40, 0.41] | 0.068 [0.04, 0.09] |
| B: Decider 2B text, then E4B text, then a person | 0.178 [0.15, 0.20] | 0.79 [0.76, 0.82] | 0.004 [0.00, 0.01] |
| C: Gemma e2b, then Gemma e4b, own confidence, then a person | 0.675 [0.65, 0.70] | 0.00 [0.00, 0.00] | 0.325 [0.30, 0.35] |

Findings to carry into the text:

- The two frame-reading decision models are the only calibrated arms (calibration error 0.07 and 0.08; every Gemma arm 0.19 or worse). The 2B is never confidently wrong on 676 decisions; the 27B buys 2.5 points of accuracy for 1.6 percent confidently wrong.
- A ladder with no vision-language model in it ties the best Gemma ladder: same share decided alone (0.79), same hand-off (0.13), fewer confidently-wrong answers (0.6 against 0.9 percent, intervals overlapping). The second rung's value is on in_bin: the 2B alone hands off 79 percent of in_bin decisions, and either second rung takes nearly all of them and gets 96 percent right.
- On the decisions the 2B escalates, the 27B is right 76 percent of the time against the 2B's own guess at 58 percent, so the escalation earns its latency.
- Nobody earned trust on sort_complete (91 percent handed to the person in every ladder) or on safe_to_grasp, where the hard rule does the work.
- Text-only decision models cannot do the job on their own (0.16 to 0.18 decided alone): the state text does not describe the part, so finish can only be answered from the frame or from Gemma's description of it. That is prompt design, not a verdict on the models.
- Self-stated confidence is the cautionary row: the Gemma self-confidence ladder never hands off and is confidently wrong a third of the time.
- Gemma 12b answers the cautious option on nearly every question (defective, no, abort) at probability near 1.0 while describing the same frames correctly when asked in prose: the one-word prompt with first-token probabilities does not transfer from e4b to 12b. Kept as measured.
- One prompt change on Oct 11 (naming the bin colours and sides) moved Gemma from 0.14 to 1.00 right on in_bin; prompt sensitivity is large and model-specific, as the decision-model survey literature also reports.

## Experiments still to run

- The live-ladder check: run the chosen ladder (Decider 2B, then pplx-decider 27B, then a person) for real on 20 fresh episodes and report live accuracy beside the replay prediction.
- Fine-tune the Decider 2B on logged decisions; report the share handed to a person at zero confidently wrong, before and after.
- Cloud models as arms (OpenRouter; Gemini 4 Argon for Robotics as the Gemini Robotics comparison the brief names), and the open models that fit the card but are not yet pulled (Winnow-12B, Intern-Decision 4B, d3-flash, Clef-Flash, Nimble v3).
- The compute the real robot carries: a Jetson AGX Orin 64 GB (the host, Kaggle discussion 748938, Oct 14; the robot's own stack uses 13 percent of memory and half the GPU; a box on the robot's local network also counts as on-device). The chosen ladder needs about 24 GB of GPU memory (4.7 GB for the 2B, 19.1 GB for the 27B at the benchmark's 4k context, nvidia-smi, Oct 14): more than one 3090 holds with the desktop, so the two rungs ran one at a time here and fit together only on the Orin's 64 GB. Still to do: Orin speed, and the compute-per-arm table framed for the MH4.
- Replace simulator ground truth in the safe_to_grasp rule with sensors a real robot has.
- Learning from failures: failures per episode over a long run, with and without it.
- Spoken operator answers, and asking the robot what went wrong.

## Sources

Papers whose ideas are used or tested here, with PDFs and what came of each: `docs/papers/README.md`.

- arXiv 2610.14117 (π0.5 learns to call Astra): related work. Escalation to a model only, no person rung, no comparison with a confidence threshold.
- arXiv 2610.21412 (JevAgentHarness, per-choice thresholds): tested as the joint threshold fit, Oct 16. It did not help on our data (VALIDATION.md).

## Limits to state plainly

- Simulation only. One robot, one task, block-shaped parts.
- The finish defects are painted textures; how visible they are to a person is still to be checked. Asked in prose, neither Gemma sees patches on a severity-1.0 defective part, and the 2B's only finish misses are at severity 0.1.
- 974 decisions from 90 episodes; every rate carries a 95 percent interval from resampling episodes (decisions inside an episode share parts, bins and camera, so the episode is the unit). The minority classes are thin: 60 aborts, 48 in-bin misses.
- Every recorded decision sits on the oracle's path: the arms never see a state that follows a wrong answer, so a ladder's replay score is not its live score. The check is a live run of the chosen ladder on 20 fresh episodes, reported beside the replay prediction.
- The person in the benchmark is the simulator's truth, so the person rung is assumed always right.
- Workcell cameras, not the robot's own eyes, answer the questions.
- The depth data is the simulator's: exact, no noise. Sub-millimetre bin location says the method is sound, not that a real depth camera would match it.
- Earlier logs (log02, test01, test02) were made with the bins fixed and their positions read from the scene file.

## Before publishing

- Permission from Mirsee: the licence grants `MH3.usd` for hackathon submissions only.
- Nothing is public until the hackathon closes.

## Figures to make

- Confidently wrong against share handed to a person, one point per ladder.
- Reliability curve per arm.
- Per-question bar chart of who answered.
- Made Oct 14 (`scripts/charts.py --log output/sort/test03`, PNGs in `output/sort/test03/charts/`): right and confidently-wrong rate per arm with intervals, the two per-question matrices, the ladders by decided-alone-and-right, mean time per decision.
- The same two numbers before and after fine-tuning.
