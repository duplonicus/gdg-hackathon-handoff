# Facts on screen in the intro

Every model output, number and claim the intro shows, with the file and line or key it was read from (read 2026-11-05, in the capsule worktree, branch feat/capsule-cell). Nothing goes on screen that is not in this file.

## The picture

| On screen | Source |
|---|---|
| The capsule picture | `output/capsules/sel01/frames/ep1004_438_leak_capsule.png` (110 x 110 px, copied unchanged to `assets/capsule_ep1004_438.png`, shown scaled up) |
| "A real frame from my simulator log: capsule log sel01, episode 1004, picture 438" | `output/capsules/sel01/decisions.jsonl` line 439: `"episode": "ep1004", "index": 438, "question": "leak_capsule", "frame": "frames/ep1004_438_leak_capsule.png"` |
| The question "Is the capsule in the middle of this picture a leaker?" | `src/handoff/decider.py` line 102, the `leak_capsule` question text (the exact words the models were asked) |
| "the truth, from the simulator: leaker" | `output/capsules/sel01/decisions.jsonl` line 439, `"truth": "leaker"` (sel01 was run with the simulator's truth answering; VALIDATION.md, "Capsule selection log sel01") |

## The two answers

| On screen | Source |
|---|---|
| VDM, d3 27B vision: leaker **0.98**, good **0.02** (bars drawn at 0.982 and 0.018) | `output/capsules/sel01/arms.jsonl` line 439: `"arm": "d3-27b-vision", "index_global": 438, "probs": {"leaker": 0.982064522586279, "good": 0.01793547741372111}`, truth leaker |
| VLM, Qwen3.8-27B, "asked for one word, it wrote: **good**" | `output/capsules/sel01/arms.jsonl` line 11473: `"arm": "qwen-27b-vision", "index_global": 438, ... "extra": {"text": "good"}` (the model's reply, stored as text; prompt in `src/handoff/rungs.py` `one_word_prompt`, "Answer with exactly one word") |
| VLM, Qwen3.8-27B, "asked how sure, from 0 to 1: **0.95**" | `output/capsules/sel01/arms.jsonl` line 44575: `"arm": "qwen-27b-self-vision", "index_global": 438, "probs": {"leaker": 0.05, "good": 0.95}, "extra": {"self_confidence": 0.95}`. Its answer was "good". Prompt in `src/handoff/rungs.py` `self_confidence_prompt` ("Reply in JSON with your answer and how confident you are from 0 to 1"). **The raw JSON string is not stored**, only the parsed answer and confidence, so the screen shows the two values, not a quoted reply. |
| "Same 27B base: d3 is built on Qwen3.8-27B" | `src/handoff/rungs.py`, the Qwen rung docstring: "Qwen3.8-27B, the model d3 27B is built on: dup, Nov 4"; VALIDATION.md, "First replays on the capsule selection log": "the VLM same backbone as d3" |
| VLM "reads the picture, writes words"; VDM "reads the picture, gives each option a probability" | WRITEUP.md, "Vision decision models versus vision-language models" ("A VLM generates text", "A VDM ... returns a probability for each one") |
| "vision decision model" | WRITEUP.md: "vision decision models (VDMs, my shorthand ...)" |

Not shown, but true of the same picture and available if dup wants them: Qwen's first-word probability for "good" was 0.956 (line 11473), Intern-Decision-4B said leaker 0.971 (line 22507), Decider 2B said leaker 0.932 (line 33541).

## How the picture was picked (said on screen as "picked to show the difference")

Method: over `decisions.jsonl` and `arms.jsonl`, the `leak_capsule` questions whose truth is "leaker" (282 of them), keep those where d3 said leaker at 0.9 or more AND Qwen's one-word arm said good at 0.9 or more: **67 pictures**. Picture 438 is the first of them by index whose pool is plainly visible at a glance (checked by eye on a contact sheet of six). It is a picked example, not a typical one, and the footnote says so.

## The footnote numbers (EXPLORATORY)

| On screen | Source |
|---|---|
| "Across all 595 leaker pictures in this log, d3 caught 92%; Qwen3.8-27B caught 43% (one word) and 66% (stated)" | VALIDATION.md, "First replays on the capsule selection log (Nov 5 morning; EXPLORATORY: selection data, nothing frozen)", the "Relabelled (Nov 5, late morning)" table: leakers caught d3 27B **92%**, Qwen3.8-27B first word **43%**, Qwen3.8-27B stated **66%**, over 595 leaker pictures |
| "Exploratory: selection data, nothing frozen" | the same section's heading and last line ("Still exploratory (selection data)") |

## The terminal (dup's joke, not a measurement)

| On screen | Source |
|---|---|
| `/goal win the hackathon`, "Confabulating...", "Winning..." | dup's scene, `DUPS_VISION_SLIDES.md` (main checkout) |
| `cwd: /mnt/c/Users/dup/dev.win/GDG-Hackathon` | the hackathon folder's real WSL path |
| "Welcome to Claude Code" | plain text in the hand-drawn terminal; no logo, no spinner glyph artwork (a plain `*` that turns) |

## After the intro (SCRIPT.md beats 1 to 3, added Nov 6)

| On screen | Source |
|---|---|
| "Maybe decision models would be well suited for robotics? After all, aren't robots just making a series of decisions?" and "my question at the hackathon's launch meeting" | dup's own words (his notes at the bottom of `../film/SCRIPT.md`, Nov 6); the meeting is the Oct launch, where the hosts read out a decision-model idea from "Jacob" (`state/HISTORY.md`, launch meeting notes) |
| "Jev · Sep 15" | Jev (TypeSafe) launched September 15, 2026 (web search Nov 6; OpenRouter lists `typesafe/jev-1.13-20260917`) |
| The 28 names raining down (Decider 2B, d3 27B, Intern-Decision, StartLux, Winnow-12B, Nimble 9B, Vega-27B, Lux-9B, Zyot Decider, decisio, Jeff-1.1, Sieve-31B, Wald 4B, Elephant-1, Mammoth-1, Quyet 1.0, Lev-3B, Vela 3.0, GLiDE, KnowLine, Chon-1.0, Drex Max, pplx-decider, d1-2B, Torchcast, bekko, Matilda, Von-1.0) and "Names from my decision-model watch log, Sep 15 to Nov 6, 2026" | `state/HISTORY.md`, the decision-model watch entries (names shortened to fit a box; the order they fall in is not their release order) |
| The ladder: 2B, 4B, 9B, 27B, then a person; "acted wrong too often"; "It came down to one: d3 27B, then a person"; the footnote | VALIDATION.md, "Exploratory: small-model-first cascades on `test03`" (cascades halved hand-offs, acted wrong more often, lost at human_cost 0.2) and "Confirmation on `test04`" (the frozen ladder: d3 27B then a person) |

## The comparison as a Phase 1 finished part (Nov 6 night; placement test: after the cloud, before the ladder)

| On screen | Source |
|---|---|
| The picture, "sort log test03, episode 456, decision 606", truth "good" | `output/sort/test03/frames/ep0456_606_finish.png` (copied to `assets/finish_ep0456_606.png`); decision 169 of `output/sort/test03_finish/decisions.jsonl`, truth good (`good@0.00`) |
| Qwen3.8-27B: "defective", "0.95" | `test03_finish/arms.jsonl`: qwen-27b-vision defective 0.954 (first word), qwen-27b-self-vision defective 0.95 (stated) |
| d3 27B vision: good 0.94, defective 0.06 | `test03_finish/arms.jsonl`: d3-27b-vision good 0.9407 (from test03's own replay) |
| Footnote: right 93% / 94 to 95%; sure and wrong at 0.9+: d3 once, Qwen 6 to 12 times | VALIDATION.md, "Qwen3.8-27B on the finish question, test03" |

The capsule version (sel01 picture 438, VLM "good 0.95", VDM leaker 0.98) moves to Phase 2 (`../film/SCRIPT.md` beat 10); its rows above still hold for it.
