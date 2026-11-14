# Kaggle film: brief

The attached public video for the Kaggle entry (GDG Windsor "Build with AI", due Nov 13). Also the base for the Nov 20 pitch.

## The brief

- **Length:** 3:28 (beats in `SCRIPT.md`; the cap given was 2.5 to 3.5 minutes).
- **Format:** 16:9, 1920x1080, 60 fps for the drawn parts; footage at its own rate.
- **Promise in one line:** a robot that acts alone when it is sure, asks a person when it is not, and takes the safe answer when nobody answers.
- **Audience:** the judges, industry and academic (GDG Windsor, DataRealm, partners; Steven Rice is an automation specialist). They score on four lines (`state/kaggle-overview.md`): handoff architecture 30, on-device 25, impact and execution 25, writeup and video 20. Criterion 4 asks for the system end to end **including edge cases, faults and unexpected inputs**, so the film puts three faults on camera (an unsure model, a wrong-tray drop, a model that dies plus an operator who never answers) and opens on a fourth (a model that is sure and wrong).
- **Hook (0 to 2.5 s):** the real frame d3 got wrong, and the question over it.
- **Reference style, named:** David Shapiro's "Why people HATE automation" (Oct 10, 2026) whiteboard look, which dup picked on Oct 15 ("your projector argument is valid"): white paper, hand-drawn strokes that draw themselves in, three accent colours, a zoom-out ending that shows every scene at once. Changed for dup's technique (Oct 27): instead of one big canvas with a panning camera, **each scene is a slide, and every motion shot is slide n turning into slide n+1** (a magic move: what is in both moves, what is new draws in, what is gone fades). The zoom-out ending is kept: the last move pulls back from slide 11 to a grid of all the slides.

## Style decisions

- **Type:** Patrick Hand for everything drawn (OFL, already in the repo from the ladder test). Caveat is out (dup, Oct 15: "wtf"). Real log lines, numbers from the logs and sources in a monospace (Ubuntu Mono, a system font in WSL where the renders run; Consolas elsewhere).
- **Colour:** ink #2b2b2b on paper #fbfaf5; green = the good tray and "acted alone, right"; orange = the rework tray and d3; blue = the parts and the person; red = wrong. One meaning per colour across the film.
- **The robot is drawn as the MH3 really is:** a humanoid torso on a wheeled base, no legs.
- **Real pictures only from the real system:** the hook frame is test04's own camera frame (`output/sort/test04/frames/ep1009_104_in_bin.png`). The operator page and the viewer appear only as screen recordings, never redrawn as fake screens.
- **Sources on screen:** every slide with a number carries its VALIDATION.md section in small grey type (`facts.md` lists them all). The caveats are on screen too: load cells not in test04, judgement questions only, simulated parts.
- **Motion:** cubic in-out moves, things leave in the first 45% of a move, new things draw in after; shared elements ride on top; a 1.2% camera push during each hold so no slide sits dead still; the builds on each slide keep drawing through the hold, timed to the narration.
- **Engine:** `film.js` (one `seek(t)` engine, seeded wobble, no timers) with three entry pages: `film.html` (the whole film as an animatic, grey cards where footage goes), `slides.html` (one second per slide, fully built: the stills), `shot_04_05.html` (the proof shot). Render with `~/dev/motion/.venv/bin/python ~/dev/motion/scripts/render.py <page> probe|draft|full`.
- **Voice:** dup's own voice, in his words: the narration in SCRIPT.md is a draft to say his way, first person singular (solo entry). No music needed; a quiet bed ducked under the voice if wanted.

## What exists (Oct 27)

- `SCRIPT.md`: 17 beats, narration draft, rubric line per beat, the shot list with exact commands, assembly steps.
- `facts.md`: every number with its source line.
- `stills/slide_01.png` to `slide_12.png`: the 12 keyframes, rendered and checked by eye.
- `out/shot_04_05.mp4` (gitignored): the proof shot, slide 4 turning into slide 5, 5.0 s, 60 fps, motion blur.

## What is left

1. **dup:** OK the beats and the slides; put the narration in his own words.
2. **Record** shots 1 and 2 (GPU runs, each waits for dup's yes; see SCRIPT.md "Shot list to record"). Shot 1 doubles as the check that the demo runs with the network off: not verified yet.
3. After recording: put the run.log numbers into beats 8 and 11, and check beat 13's last sentence against what the robot did.
4. Re-render the stills once more if any slide changes (slide 12's grid is built from them), then `film.html full` in chunks, critique pass (contact sheets, a read-only agent restates the message from frames alone), then cut in the footage, voice, captions.
5. Not done: no full-film render, no sound, no captions.

## Proof shot check (Oct 27)

`out/shot_04_05.mp4`: 5.0 s, 1920x1080, 60 fps, 8 subframes. First render: the four question cards flew straight across the "Behaviour tree" title for about half a second. **Fixed (Oct 27, coordinator's request):** on slide 5 the cards and the robot move first, straight into the judgement box (which draws at 0.05 s), and the tree box only starts drawing at 1.45 s, after the last card has passed it; checked on 12 probe frames from 0.8 to 4.9 s and on the re-render's frames: no text over text. Still to consider: the tree box's grey tint fades in a beat before its outline draws.
