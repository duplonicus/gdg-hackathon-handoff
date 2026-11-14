# Kaggle intro: brief and beat map

A new opening for the Kaggle video (GDG Windsor "Build with AI", due Nov 13), drafted from dup's scene in `DUPS_VISION_SLIDES.md`. It stops at "Which is better?"; what comes after is not decided yet.

## The brief

- **Length:** 39.8 s. **Format:** 16:9, 1920 x 1080, drawn at 60 fps (the draft render is 30 fps, 960 wide).
- **Promise in one line:** the same picture, the same question, two kinds of model: one writes a word, the other scores each option.
- **Spine:** VDM against VLM. d3 27B vision (a VDM) is built on Qwen3.8-27B (a VLM), so the pair is the cleanest comparison in the project: same base model, different way of answering.
- **Audience:** the judges (industry and academic), as for the main film.
- **Hook (0 to 2.5 s):** the room draws itself in while dup walks on: black hoodie, aviators.
- **Reference style, named:** the main film's own (`../film/BRIEF.md`): David Shapiro's "Why people HATE automation" whiteboard look. Patrick Hand for drawn text, Ubuntu Mono for real outputs and sources, ink #2b2b2b on paper #fbfaf5, orange = d3, green = right, red = wrong, strokes that draw themselves in.
- **The terminal** is dup's own: Ubuntu aubergine (#300a24) background, Ubuntu Mono, drawn in the film's hand style (wobbly box and rules). Text only: "Welcome to Claude Code", the real cwd, the typed `/goal win the hackathon`, and dup's spinner joke ("Confabulating...", "Winning..."). No logo, no product artwork; the spinner is a plain `*` that turns.
- **Real pictures only from the real system:** the capsule picture is the real 110 px frame from `output/capsules/sel01/frames/`, scaled up, not redrawn.
- **Sound:** synthesized key clicks on every typed letter (cue times read from the page), a heavier Enter, a rising whoosh through the dive (`make_sound.py`, nothing downloaded). No music. dup's voice-over to come; the clicks are a placeholder level (peak -3 dBFS, not loudness-normalised).

## Beat map (seconds)

No music, so no BPM grid: beats are set for a voice-over to land on. Nothing holds still for more than about a second before the final card (a 1.2 to 3% camera push runs through every hold).

| # | Time | What happens | Suggested VO (for dup to rewrite in his own words) |
|---|---|---|---|
| 1 | 0.0 to 3.3 | The room draws itself in (floor, chair, window, desk, monitor). dup walks on from the left, hood up, aviators on. | (none, or) "This is me." |
| 2 | 3.3 to 4.9 | He sits, leans in, hands go to the keyboard. | "This is my entry for the Build with AI hackathon." |
| 3 | 4.7 to 7.4 | The camera pans up and in to the monitor: the terminal, Ubuntu purple, the hackathon folder. | (let it breathe) |
| 4 | 7.7 to 11.4 | `/goal win the hackathon` typed letter by letter, a click per key. Enter at 11.4. | (none: let the typing land the joke) |
| 5 | 11.7 to 14.3 | `* Confabulating...`, then at 13.4 `* Winning...` | (none) |
| 6 | 14.3 to 16.6 | The camera dives through the hole in the "g" of "Winning"; the capsule picture shows through the hole and the hole opens onto the paper. | (whoosh) |
| 7 | 16.6 to 19.6 | The picture, and the question the robot asks: "Is the capsule in the middle of this picture a leaker?" Source line under it. | "In my simulator, a robot inspects soft-gel capsules. For each one, it asks a model this question." |
| 8 | 19.6 to 24.0 | The picture shrinks to the middle. Left box draws: **VLM**, vision-language model, reads the picture, writes words, Qwen3.8-27B. It writes `good` (22.0). Asked how sure: `0.95` (23.6). | "A vision-language model reads the picture and writes an answer. This one wrote: good. Asked how sure, it said 0.95." |
| 9 | 24.8 to 28.0 | Right box draws: **VDM**, vision decision model, reads the picture, gives each option a probability, d3 27B vision. "Same 27B base: d3 is built on Qwen3.8-27B" (26.0). The bars fill: leaker 0.98, good 0.02 (26.8 to 27.8). | "A vision decision model writes nothing. It scores each option. Same base model, built on that Qwen. Leaker: 0.98." |
| 10 | 30.2 to 33.6 | The truth from the simulator: **leaker** (30.2). Red box round Qwen's `good` (31.1), green box round d3's leaker row (31.7). Footnote: one picture, picked; across 595 leaker pictures d3 caught 92%, Qwen 43% (one word) and 66% (stated), exploratory. | "It was a leaker. Same picture, same base model, two very different answers." |
| 11 | 34.2 to 39.8 | The boxes shrink to the bottom, the picture between them; **"Which is better?"** writes itself in (35.2) and holds to the end. | "So which is better?" |

## Layers (back to front)

1. `#world`: the room (floor, window, chair, desk, keyboard), the terminal `#term` inside the monitor, the monitor bezel, dup (an SVG rig: hip, torso, head, two arms, two legs; walk cycle, sit, two-link reach to the keys, a small lift on each keystroke).
2. `#paper`: the paper scene. During the dive it is masked to the shape of the g's hole (found by flood-filling the glyph on a canvas at load); after the dive the mask is off and `#world` is hidden.
3. `#gtop`: a copy of the "Winning" line with only the g visible, above the paper during the dive, so the glyph's stroke hides the mask's pixel edge at 200x zoom.

## Files

- `intro.html` + `intro.js`: the page; `window.seek(t)` draws any instant. `hud.js` for live preview (Space, arrows, `?t=`). `?debug` marks the hole's centre.
- `facts.md`: every output and number on screen with its file and line.
- `make_sound.py`: the sound effects, and the mux.
- `stills/`: the key beats. `out/` (gitignored): the draft render.
- `assets/`: Patrick Hand (from `../film/assets`) and the capsule frame.

## Render

From WSL: `~/dev/motion/.venv/bin/python ~/dev/motion/scripts/render.py intro.html probe|draft|full|contact`. **Gotcha found Nov 5:** the motion venv's Playwright 1.62 wants Chromium build 1234, and only 1228 and 1248 are in `~/.cache/ms-playwright`. These renders ran with `PLAYWRIGHT_BROWSERS_PATH` pointed at a scratch folder holding symlinks named `*-1234` to the cached 1248 builds (no download). Render output looked right; the version mismatch itself was not otherwise tested.

## What is left

1. dup: OK the beats; rewrite the VO lines in his own words; decide whether beat 10 (the truth and the red/green boxes) stays, or the intro stops on the two answers and leaves "Which is better?" open.
2. Full render (60 fps, 8 subframes, in chunks: about 40 to 60 min of CPU), then voice, then the mix (loudnorm to -14 LUFS with the VO).
3. The fonts: Ubuntu Mono is a system font in WSL where the renders run; a Windows render would need it installed.

## Critique pass (Nov 5, on the draft's frames; done by me, not by a separate agent)

Scored from tiled frames of `out/intro_draft.mp4` (one every 1.5 s, plus 5 a second through the dive) and full-size probes:

| Line | Score | Note |
|---|---|---|
| Hook, first 2 s | 7 | the room draws in and dup walks on; nothing about VLM vs VDM yet (by design: dup's scene) |
| Readability at 360 px | 6 | "good", "0.95", the bars and "Which is better?" read; the boxes' small grey labels do not |
| Motion | not scored | the walk, the sit and the dive were judged on stills only; **the draft has not been watched as video** |
| Variety | 7 | the terminal stretch (7.4 to 14.3 s) changes only by typing and the spinner |
| Composition | 8 | after the fixes below |
| Accuracy | 9 | every output traced in `facts.md`; the picked example is labelled as picked |
| Sound sync | not checked by ear | clicks are placed from the page's own key times, so they line up by construction |

Fixed after the first draft: values spilling past the VDM box, ellipse marks cutting through text (now rounded boxes), arrows poking into the boxes, a three-line footnote, the boxes' tint fading in as a flat grey block before the outline, the final card's boxes too small to read (0.5 to 0.6 scale), jagged mask edges in the dive (a copy of the g now sits on top).

Still weak: the dive's match-cut. The plan was for the leaker's pool to fill the g's hole; in the render the whole picture appears as a small framed square inside the hole, then the hole opens round it. It reads as "through the g into the picture", not as "the g's hole becomes the pool".

`render.py pops` on the 30 fps draft: one pop at frame 343 (11.43 s; the tool prints times as if 60 fps) is the terminal jumping on Enter, intended; no one-frame flashes; the longest still stretch (1.13 s) ends at 38.7 s, inside the final hold.
