# Capsule cell: what freezes before the one GPU night

dup, Nov 5: "let's make sure everything is good from now on before wasting an entire night of gpu time." The capsule cell changed after the first selection log (sel01: 260 packed capsules, one hand, driving). Everything below changes the pictures, the questions or the scoring, so it is settled before the new selection log is made, and then it does not move. Each line shows its default; silence means the default.

## 1. The scene (settled; built Nov 5 to 6)

- **Layout:** two half-trays, 4 rows x 4 columns a side, 32 capsules, pitch 65 x 25 mm, a 15 cm aisle down the middle. Every spot is inside the reach measured palm down from one park (VALIDATION, "Two half-trays inside the measured reach").
- **Robot:** parks once per shift at the cell, base braked; the hand on the capsule's side picks it with a palm-down pinch (welded to the index fingertip) and drops it in that side's leaker bin.
- **Cameras the models see:** fixed scene cameras only. Tray camera 1920 x 1200 from above the far side; one camera over each leaker bin (before and after pictures); one overview per cell for "where am I needed". The robot's head and base cameras are off in capsule runs.
- **Leakers:** 0 to 5 a shift, pool 16, 22 or 30 mm across; a leaker differs from a good capsule only by its pool.

## 2. The questions (settled)

- where_needed (left / right / neither), leak_block (yes / no), leak_capsule (leaker / good), leaker_landed (yes / no), tray_clear (yes / no). Their scene text changed on Nov 5 to say "two red leaker bins beside the tray".
- **Block size: 16** (a whole half-tray). Default: keep. A block's picture can no longer show another block's leaker (the aisle), so the 61 neighbour pictures of sel01 do not arise.
- A leak_block picture with a leaker only partly in view is labelled unclear and left out of scoring (rule of e5ef177). On this layout that happens only at a crop's edge in the block-size cuts.

## 3. The selection log (the GPU-free half of the night)

- **sel03 replaced sel02** (Nov 6 to 7 night) after the lighting, drop fix and tile floor changed the pictures; same seeds and commands, `--out output/capsules/sel03` (VALIDATION).
- **sel02:** `run_capsules.py --episodes 150 --seed0 2000`, oracle answers, lean headless (about 1.4x real time: roughly 2 hours). Seeds 2000 to 2149, none used before.
- **Block-size cuts** of its first 20 trays: sizes 1, 4, 16 and the whole tray (64 is gone: a half-tray holds 16).

## 4. The replays (the GPU half)

- **Arms, in order:** d3-27b-vision, intern-4b-vision, qwen-27b-vision (first word), qwen-27b-self-vision (stated confidence). **Default: drop decider-2b-vision**: on sel01 it called nearly every capsule a leaker (VALIDATION).
- Gemma e4b is not wired into `replay_capsules.py`. Default: leave it out of the capsule study.
- One model on the GPU at a time, VRAM and commit gated, resumable (`replay_capsules.py`).

## 5. Then, once, on fresh data

- **PREREG v2** written and committed from sel03's numbers (Nov 7, `docs/PREREG_CAPSULES.md`): which arm, which threshold, what counts as caught. leaker_landed replaces in_bin.
- **test05:** `run_capsules.py --episodes 150 --seed0 3000`, never looked at before scoring; scored once.

## Open (not blocking the night)

- The head: the fixed tray camera does the seeing. Default: tilt the head down at the tray while working, for looks.
- A leaker that misses a bin and is not back on the tray fails the shift; the robot does not yet retry or call a person for it. A drop caught the rim once in 9 (it bounced back and was re-picked).
- Viewer pace 0.66x real time; headless 1.35 to 1.53x.
