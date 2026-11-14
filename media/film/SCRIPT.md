# Kaggle film: script (v2, Nov 6)

**About 4:04**. Kaggle sets no length limit (Overview and Rules read Nov 6); aim under about 4 minutes, 16:9, 1920x1080, one video: the intro (`../intro/`) joined on the front. v1's beats (Oct 27, one study) are in git history; v2 keeps Phase 1 and adds Phase 2, as the README does.

Narration is a **draft in my voice for dup to rewrite in his own words** (solo entry: I and my, never we). About 150 words a minute. Every number is in `facts.md` with its VALIDATION.md section; Phase 2 numbers wait for test05 and are marked **[test05]**.

Rubric: **1** Architecture, deterministic logic and AI handoffs (30) · **2** Shop-floor deployment, on-device (25) · **3** Industrial impact and execution (25) · **4** Writeup, demo, justification (20; the video shows the system end to end, with edge cases, faults and unexpected inputs).

Settled with dup (Nov 6): his notes at the bottom of this file set the content, and his order is intro, his question, the cloud, a Phase 1 title card, the finished-part comparison (no leakers until Phase 2), the ladder, and the dive through the g lands in his thought bubble with the VLM/VDM comparison moved to Phase 2 (beat 10) (folded in below); the in_bin blind spot (29 of 29) is cut from the film and stays in the writeup, since explaining why and the fix takes two frames; Phase 1 stays as tested, Phase 2 added after it; the closing pull-back shows every slide **with no borders round them**; human-in-the-loop shots are simulated (scripted answers on the operator page) and recorded, not played live; the VO is made with ElevenLabs (Flash for drafts, v4 or v3 for the final).

## Beats

| # | Time | On screen | Narration (draft; beat 1 is dup's own words) | Footage | Rubric |
|---|---|---|---|---|---|
| 0 | 0:00 to 0:17 | **INTRO** (built): dup walks on and sits, `/goal win the hackathon`, "Confabulating...", "Winning...", and the camera dives through the hole in the g **straight into his thought bubble** (dup, Nov 6: the VLM/VDM comparison moved to beat 10). | (none: footsteps, the typing, the whoosh) | none | 3 |
| 1 | 0:17 to 0:26 | **THE THOUGHT BUBBLE** (new): dup, hand on chin, a thought bubble with his question. Small caption: "asked at the hackathon's launch meeting". | Maybe decision models would be well suited for robotics? After all, aren't robots just making a series of decisions? | none | 1 |
| 2 | 0:26 to 0:42 | **THE CLOUD** (new): a smiling cloud in our whiteboard style with two hugging hands, so it reads as Hugging Face (our drawing, not their logo). Jev drops out first, labelled "Jev · Sep 15"; then more and more decision models rain down on dup in his hoodie, bounce off his head and pile up at his feet (real names from the watch log: Decider, d3, Intern-Decision, StartLux, Winnow, Nimble...). | Decision models are brand new. Instead of writing an answer, they score every option. Jev came out in mid-September. By October, they were raining down. | none | 3 |
| 3 | 0:42 to 0:49 | **PHASE 1 TITLE CARD** (built): "Phase 1: Proof of concept", under it "Painted part inspection"; slide 3's hand-drawn MH3 rolls in from the left on its wheeled base, then the table, a part and the good and rework trays draw in. | Phase one, a proof of concept: inspecting painted parts. | none | 3 |
| 4 | 0:49 to 1:09 | **SAME PICTURE, TWO ANSWERS: A FINISHED PART** (built): "Is the finish on this part good or defective?"; a real test03 frame (episode 456, decision 606), "The dark patches are shadows, not paint."; the VLM (Qwen3.8-27B) wrote "defective", 0.95 sure; the VDM (d3 27B, built on Qwen) scored good 0.94; the truth: good. Footnote: across 270 finish pictures both are right about as often, but sure and wrong d3 once, Qwen 6 to 12 times. | Here's the difference. Same picture, same base model. Qwen writes an answer: defective, 95 percent sure. d3 scores both options: good, 94. The part was fine. Across all my test pictures they're right about as often, but Qwen is sure and wrong far more. | none | 1 |
| 5 | 1:09 to 1:27 | **THE LADDER I TRIED FIRST** (new): a ladder of small to large models, 2B, 4B, 9B, 27B; the small rungs flash red ("acted wrong"); they fall away, leaving d3 27B, then a person. | So what's the best result I could get? My first idea was a ladder: small models first, bigger ones only when unsure. On my selection runs, the small ones were wrong too often. It came down to one: d3 27B, then a person. | none | 1, 4 |
| 6 | 1:27 to 1:39 | **THE CELL** (the title card's drawing, now with footage): the MH3 on its base, the table, a painted part, green and orange trays. Then footage. | A Mirsee MH3 humanoid in the Genesis simulator, sorting painted parts. Good ones go green, defective ones orange, for rework. | **P1-A**: one whole sort cycle (part_12: reach, grasp, carry, rework bin, arm home; 16 s) | 3, 4 |
| 7 | 1:39 to 1:57 | **SLIDE: the split** (v1 slides 4 and 5 merged): the four questions fly into "Judgement"; the robot sits in "Behaviour tree". | The work is split. A behaviour tree runs everything that must never fail: motion, grip, safety. Four questions it can't script go to the model: is the finish bad, safe to grasp, right tray, sort done? The model only answers. It never moves the arm. | none | 1 |
| 8 | 1:57 to 2:13 | **SLIDE: the ladder at work** (v1 slide 6): d3 27B vision, the 0.80 mark, a person on the operator page, the 120 s clock, the safe answers. Then footage. | At 80 percent sure or more, the robot acts. Below that, a person decides on the operator page. Nobody answers in two minutes? It takes the safe answer. | **P1-B**: an unsure finish call goes to the page; a scripted answer; the robot carries on | 1, 4 |
| 9 | 2:13 to 2:27 | **SLIDE: pre-registered, scored once** (v1 slide 11): 78.9% / 18.5% / 2.6% (1.7 to 3.5). | I froze the ladder before testing it, then ran it once on 300 new episodes. On the judgement calls: right on its own 78.9 percent of the time, handed 18.5 to a person, wrong 2.6. | none | 3, 4 |
| 10 | 2:27 to 2:35 | **THE BRIDGE** (new): the intro's "Which is better?" comes back. | The idea works. Now back to the question: when a model says it's sure, which kind can you trust? | none | 1 |
| 11 | 2:35 to 2:50 | **PHASE 2: THE CAPSULE CELL** (new): the tray of soft gels, a leaker's pool, the two red bins. Then footage. | Phase two: a soft-gel capsule line. A leaker looks like every other capsule, except for the pool under it. The robot parks at the tray, finds each leaker, and drops it in the bin. | **P2-A**: one shift from in-sim cameras: park, look, pinch a leaker, drop it in the bin | 3, 4 |
| 12 | 2:50 to 3:12 | **SAME PICTURE, TWO ANSWERS** (built, moved here from the intro): the capsule picture and "Is the capsule in the middle of this picture a leaker?"; the VLM (Qwen3.8-27B) writes "good · 0.95"; the VDM (d3 27B, built on Qwen) scores leaker 0.98; the truth: leaker; "Which is better?" | Same base model, two ways of answering. Qwen reads the picture and writes: good, 95 percent sure. d3 is built on Qwen, and scores every option instead: leaker, 98. It was a leaker. One picture, though. Which is better? | none | 1 |
| 13 | 3:12 to 3:26 | **SLIDE: the answer** (new): leakers missed while 0.9+ sure, per model, test05 **[test05]**. | On [test05: N] leaker pictures it had never seen, the VLM missed [N] while ninety percent sure. The decision models missed [N]. *(say what test05 shows, whichever way it falls)* | none | 1, 3 |
| 14 | 3:26 to 3:32 | the same slide, footnote: chosen on one log, frozen, scored once on another. | Same rules as phase one: picked on one log, frozen, scored once. | none | 3, 4 |
| 15 | 3:32 to 3:52 | **SLIDE: when something fails, on one PC** (v1 slides 9 and 10 merged): the model goes down, nobody answers, one RTX 3090, no cloud. Then footage. | What if something fails? Here I kill the model mid-run: the question goes straight to the page. Then nobody answers: two minutes later, sped up here, it takes the safe answer and doesn't grasp. All of it on one PC with one graphics card. No cloud. | **P1-D**: d3 stopped mid-run, a scripted answer, then no answer: the 120 s time-lapse, the safe answer | 2, 4 |
| 16 | 3:52 to 4:04 | **CLOSE** (v1 slide 12): the camera pulls back to every slide, **no borders**, three lines over them. | Logic for what must never fail. A model for what can't be scripted. And a person whenever the model isn't sure. | none | 1 |

What each rubric line gets: **1** beats 1, 4, 5, 7, 8, 10, 12, 13, 16 · **2** beat 15 (the model going down is rubric 2's "connectivity drops safely") · **3** beats 0, 2, 3, 6, 9, 11, 13, 14 · **4** footage P1-A, P1-B, P1-D and P2-A: the run end to end, an unsure model, a model that dies, an operator who never answers.

## Sound

Synthesized in code (`../intro/make_sound.py` makes the key clicks and the whoosh that way; nothing downloaded): soft taps on each of dup's footsteps (timed off the walk cycle), a pencil scratch whenever a scene draws in (timed off the page's own stroke times), a light bonk as each model hits dup's head and a soft clatter as they pile up. ElevenLabs can generate sound effects too, but on the free plan they must credit elevenlabs.io.

## Footage

Two kinds, both made by script so a shot can be redone the same way:

- **Robot-only shots (P1-A, P2-A): in-sim cameras.** The scene records its own frames at 30 a second of sim time (`watch_shift.py` does this every 1.5 s today), ffmpeg joins them. Clean frames, any angle, smooth whatever the sim's speed. Phase 1 shots on Phase 1's scene as tested (checker floor, brown table); Phase 2 on the tile floor and steel table.
- **Human-in-the-loop shots (P1-B, P1-D): simulated, not live (dup, Nov 6).** `demo.py` runs with the operator page; a Playwright script answers the page as the shot needs (a click after a few seconds, or no answer at all), and screenshots it as it goes, while the scene records its frames. The page frames and the sim frames are laid side by side by sim time, with the log panel's lines. No screen recorder: WSL's ffmpeg can't grab the Windows desktop and none is installed on Windows.
- **What the robot does on screen is what the log says it did.** The numbers said over footage (the unsure call's confidence, the load cell's grams) come from that recording's `run.log`, written in here after it is made.
- **Every speed-up gets a label** ("2x", "120 s, sped up").

Not built yet: the 30-a-second frame recorder, the Playwright answerer, the side-by-side compositor. P1-B and P1-D need d3 27B loaded (about 18 GB of GPU): GPU nights, after test05.

## Slides and motion

- Kept from v1 (`slides.html`, `film.js`): the cell, the split (merged with the four questions), the ladder, the result, the close. Dropped: v1's hook and title (the intro does their job), sure and wrong and the load cell (cut, Nov 6).
- New: the cloud, the thought bubble, the ladder I tried first, the bridge, the capsule cell, same base two answers (reuses the intro's boxes), the answer (layout now, numbers after test05).
- The close: the camera pulls back to a grid of every slide with **no borders** (v1 drew one round each).
- `film.js`'s `FILM` list still holds v1's timings; it gets v2's when the slides are rebuilt.

## Order of work

1. dup rewrites the narration in his words (this table).
2. New slides as stills; dup looks.
3. Motion between slides; the animatic (grey cards for footage, no sound).
4. Footage, after test05 (GPU).
5. VO in ElevenLabs: Flash drafts per beat, then one v4 or v3 final; captions and beat lengths from the final's timings.
6. Mix (loudnorm -14 LUFS), captions, join the intro.


# Dup's vision for script and video

- Existing intro plays. it's pretty good. but we need some tapping sound effect for the footstep. choose appropriate sound effects for the rest of it too. maybe like a pencil scribing sound effect for when the scenes first draw? chkachkachkacha
- We explain the history of decision models: started releasing in 2026 > Jev in mid september > influx during october. We can show a huggingface smiley, but it's a cloud in the sky. the models that came out during october come out of the cloud and down to me (in the black hoodie) bouncing off of my head and piling up on the ground at my feet. 
- "Maybe decision models would be well suited for robotics?" the question I asked during the intro meeting. "After all, aren't robots just making a series of decisions?" in a thought bubble with thinking face/hand on chin
- What is the best result we can get? then we talk about the naive original ladder with small > medium > large model and how it ultimately landed on d3 27 B alone. 
- Phase 1: proof of concept
- Scrap the part about the bug with the in-bin detection; that can go in the writeup and lessons.  
