# Facts on screen and in the narration

Every number the film shows or says, with where it comes from. Sources are VALIDATION.md section names (its line numbers move as sections are added); every value was re-read on 2026-10-27 after the evening's edits. No number goes on screen that is not in this file.

| Fact (as said or shown) | Source |
|---|---|
| test04: 300 episodes, 3,262 decisions, every one with a truth | VALIDATION.md, "The confirmation log, `test04`" |
| Ladder frozen in docs/PREREG.md (1c6c475) before any test04 replay; d3-27b-vision then a person, one threshold 0.80; scored once, no refit | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once" |
| Judgement questions (2,262): decided alone and right **78.9%** (77.0 to 80.9) | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once", table row "Decided alone and right" |
| Judgement questions: handed to a person **18.5%** (16.8 to 20.2) | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once", table row "Handed to the person" |
| Judgement questions: acted wrong **2.6%** (1.7 to 3.5) | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once", table row "Acted wrong" |
| Intervals are 95%, bootstrapped by episode | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once" |
| **29 of 29**: d3 answered "in the bin" on all 29 in_bin decisions about a part in the wrong tray on test04, every one at 0.9 confidence or more. Those 29 decisions are on **15 wrong-tray landings**: after a "no" the tree asks again, so most landings were checked twice | VALIDATION.md, "Where the frozen ladder hands off and acts wrong on `test04`, by fault type", the wrong-bin table and the paragraph under it (counted Oct 27 from `decisions.jsonl`) |
| The hook frame: test04 episode 1009, decision 104, a part meant for rework in the good tray, d3 "yes" at **0.99** | VALIDATION.md, "Where the frozen ladder hands off and acts wrong on `test04`, by fault type"; the frame is `output/sort/test04/frames/ep1009_104_in_bin.png` (copied to `assets/`); d3's answer re-read from `output/sort/test04/arms.jsonl` row 104: yes 0.9933, 277.4 ms, truth no |
| d3 27B: a 4-bit quant on one RTX 3090, **277 ms** per decision ("about a quarter of a second", "about 0.28 s") | VALIDATION.md, "27B frame arms as 4-bit GGUFs on `test03`", "What it shows" (test03 benchmark, split seed 0) |
| The confidence threshold is **0.80** | VALIDATION.md, "Confirmation on `test04`: the frozen ladder, scored once"; `scripts/demo.py` `FROZEN_THRESHOLD` |
| Operator timeout **120 s**, then the safe answer | `scripts/demo.py --operator-timeout` default 120.0; `src/handoff/operator.py` |
| Safe answers: finish defective, safe_to_grasp ask_operator (no grasp), in_bin no, sort_complete no | `src/handoff/decider.py` QUESTIONS |
| A rung that errors (model down) is skipped: the question climbs to the person | `src/handoff/decider.py` `timed` and `Ladder.decide` |
| Not safe to grasp after two asks: the part goes to the reject lane, the robot does not grasp | `src/handoff/tasks/sort.py` (Retry 2 on safe_to_grasp, `reject_part`) |
| Load cells answer in_bin with `--scales`; they are NOT part of the frozen ladder or test04 | VALIDATION.md, "Planned: in_bin from cell sensors, then the camera", the "Scored from the log, Oct 21" paragraph; `scripts/demo.py` docstring |
| Load cells caught every forced wrong-bin landing tried in Genesis (2 of 2 in `scale-wrongbin`, plus the Oct 21 and Oct 23 demos) | VALIDATION.md, "Planned: in_bin from cell sensors, then the camera" ("Checked:" paragraph), and "Measured: the touch grip against the fixed closure, seeds 1 to 10" (the Oct 23 demo paragraph) |
| Everything runs on one PC: d3 on 127.0.0.1:8774, the operator page on 127.0.0.1:8790 | `src/handoff/servers.py` D3_URL; `scripts/demo.py` OperatorPage(port=8790) |
| Jev (TypeSafe), the first decision model in the story: launched **September 15, 2026**; OpenRouter lists Jev 1.13 on Sep 17 | web search Nov 6 (launch coverage dates it Sep 15; OpenRouter `typesafe/jev-1.13-20260917`) |
| The cloud's other models (Intern-Decision Sep 26, Winnow-12B Sep 20, StartLux Sep 29, Nimble v3 Oct 13, d3, Decider...) | `state/HISTORY.md` decision-model watch entries (dates as logged there) |
| The small-first ladder lost: 2B > 4B > 9B > 27B cascades halved hand-offs but acted wrong more often; d3 27B alone won at human_cost 0.2 | VALIDATION.md, "Exploratory: small-model-first cascades on `test03`" (selection data: say "on my selection runs") |
| "No network": **NOT VERIFIED** (shot E was never recorded). The v2 slide says "no cloud calls" instead, which is true: no cloud model is in the loop | WRITEUP.md section 2; `scripts/demo.py` (models on 127.0.0.1) |

Footage numbers (the 0.58 in shot B, the gram readings in shot C) come from the recording's own `run.log` and are written into SCRIPT.md after the recording, not before.

## v2 additions (Nov 9): Phase 2 slides and footage

| Fact (as shown) | Source |
|---|---|
| Capsule comparison picture: sel01 episode 1004, picture 438; Qwen3.8-27B wrote "good", stated 0.95; d3 27B scored leaker 0.98, good 0.02; truth leaker; a picked picture | `media/intro/facts.md`, "The picture" and "The two answers" (same picture and values the intro was built from) |
| test05: 150 new shifts, 467 leaker pictures on the leaker questions (leak_block, leak_capsule) | VALIDATION.md, "test05: the capsule confirmation", the every-arm table |
| Leakers called good at 0.9 or more: d3 27B **1**, Intern-Decision-4B **1**, Qwen3.8-27B first word **98**, Qwen3.8-27B stated **171** | same table, "Leakers called good at 0.9 or more" column (`scripts/capsule_arm_report.py`, log `output/capsules/test05_arm_report.log` in the capsule worktree) |
| The frozen ladder (Intern-4B, then d3 27B, then a person) acted wrong on **0.2%** of **3,785** decisions; predicted **0.5%** | VALIDATION.md, "test05: the capsule confirmation", the frozen-ladder list; prediction from `docs/PREREG_CAPSULES.md` |
| Picked on sel03, frozen, scored once on test05 | `docs/PREREG_CAPSULES.md` (committed 0981889 before test05 existed) |
| Footage A, B, C, D: what happens in each clip and the numbers (d3 "defective" 0.58; the 120 s timeout to the safe answer "defective") | `media/film/FOOTAGE.md`, from each take's `run.log` |

