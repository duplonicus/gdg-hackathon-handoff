# Lessons

What this project taught us the hard way. One entry per lesson: the rule, what happened, what it cost. Newest at the bottom of each section. Add an entry the day the lesson is learned, with the numbers.

## Environment

**Do not run the Genesis viewer through WSLg.** Run the sim on native Windows (or bare Linux).
The viewer ran at 12 steps a second under WSL2 against 256 headless. A day went into ruling out the camera feeds, the control panel, GPU sharing with the models, vsync, shadows and the vertex count; the cost was the per-draw-call time of Genesis's renderer going through Mesa. The same scene on native Windows: 97 steps a second with the full robot, measured the evening we moved (Oct 11). The workaround we built first (cutting the robot's meshes to a quarter) reached 74 and made the robot look worse.

**Measure the slow thing before fixing it.**
Three wrong causes were announced for the lag before a profile was taken. The profile took ten minutes and named the cause.

**On Windows, call services by `127.0.0.1`, not `localhost`.**
`localhost` resolves to `::1` first and a service running in WSL or Docker never answers there: Ollama timed out at 8 s on `localhost` and answered in 75 ms on `127.0.0.1`.

**On Windows, always launch through `run.ps1`.**
A plain `uv run` re-syncs against the lock file and swaps the CUDA build of torch for the CPU one, silently.

**Open text files with `encoding="utf-8"`.**
Windows defaults to cp1252, so logs written on Linux would be misread there.

**The first build after a scene change is slow on Windows.**
257 s cold against about 40 s cached. Plan test loops around it.

## The robot

**The MH3 hand is a pincher, not a wrapper.**
The fingers are 5 cm long on a palm 7 cm thick, so a 5 cm block never fits inside the curl (3 to 4 of 10 at any stiffness). A 4 cm part, finger stiffness 5 instead of the kit's 350, and the hand yawed to match the part gave 10 of 10.

**Stiffer fingers are worse.**
At the kit's stiffness the contacts carry 170 to 200 N and jitter the part out of the grip. Force control on the finger roots was worse still (0 of 10).

**The base rolls freely until it is told to stop.**
The arm pushing a box rolled the robot 2.5 m backwards. Issue a stop after every build and reset.

**Check a joint's sign on a camera frame, not in a comment.**
`head_yaw` is positive to the robot's right. A comment said left; the first head-tracking code turned the head away from the work.

**Ask what the robot is being told for free.**
The bins sat at slightly different distances and nothing went wrong, because the robot read both positions from the scene file. It took a person looking at the scene to notice. Now the bins move each episode and the robot finds them from the camera (within 1 mm in sim). The same question is still open for the pick station, the part's pose and the safe-to-grasp rule.

**Know which camera each question uses.**
The head went level after the first part and nothing broke, because no question used the head camera: the finish question uses a fixed camera over the pick station and the bin questions a fixed camera over the bins. It looked wrong in the demo for a day before anyone asked.

**Let go slowly.**
Snapping the hand open flicked the part 8 to 9 cm away from the robot. From the third spot in a bin that put it over the far wall 6 times in 6, and the recovery aimed at the same spot and failed the same way, so the sort fully succeeded in only 13 of 20 clean episodes. Opening the hand over one second: 20 of 20 on the same seeds (Oct 12).

**The return to home between parts costs about 6% and buys nothing we could measure.**
Twenty episodes each way with the slow release: 52.3 s with it, 49.4 s without, 20 and 19 of 20 succeeding. It stays on for now only because without it the hand is still near the bins when the in-bin camera frame is taken, and the models have not been tested on those frames.

**Gravity compensation costs 4.8x in speed.**
90 steps a second with it, 430 without, and the arms droop only 1.4 mm in 3 s without it.

## Models and decisions

**A bigger model is not a better rung.**
On the test set the 2B Decider looking at the frame beat both larger text-only Deciders, and the small Gemma was confidently wrong 52% of the time at a mean confidence of 0.99. Each rung has to be measured on the questions escalated to it.

**The prompt can matter more than the model.**
Naming the bin colours and sides in the state text took Gemma from 0.14 to 1.00 right on the in-bin question.

**Do not pick thresholds on the data you report.**
Thresholds and the prompt were first chosen on the same 24 episodes; those numbers were not reported as final. A fresh test set followed.

**A label can be wrong even when it comes from the simulator.**
A part resting against a tray wall reported a steady speed of 0.06 to 0.45 m/s without moving, so a speed-based "is it in the bin" check said no on placed parts and the robot tried to recover them. Bin checks are now geometry only, and rest is judged by position over time.

**A JSON schema breaks Ollama's log-probabilities.**
Option probabilities come from a free one-word answer instead.

**A model that is down must not look like a model that is unsure.**
In the first real-ladder run on Windows the Decider server was not running. Every one of its 11 answers was a connection error, logged as an even split across the options, so every question climbed and the operator answered 8 of 11. It looked like the ladder asking at every step; it was a missing rung. The demo now checks its model servers before it builds and stops with the start command (Oct 12).

**Give every question a safe option.**
The finish question had none, so an unanswered operator prompt sent an unpainted part to the good bin.

**Ask only what changes what the robot does.**
The finish question had three answers (good, scratched, unpainted) and the robot had two bins. Scored three ways, Gemma was marked wrong for calling a grey-patch part "scratched" while sending it to the right bin, which hid that the Decider's real mistake was calling those parts good. A person looking at the frames could not tell the two defects apart either, and neither name fitted what was drawn. Cut to good or defective (Oct 12): Gemma alone went from 73% to 82% right on the same answers, and a third fewer questions reached a person. The finer label stays in the log, and can be a second question asked only about parts already judged defective.

**Look at the pictures the model gets before reading its score.**
"Finish is near chance for every model" stood for a day. Four frames and a table of what each model answered for each true finish showed scratches were caught 10 times in 10 and one kind of part was the whole problem. Showing those frames to a person was the first item in the stage 3 plan and had not been done.

## Working method

**Record once, replay through every model.**
Decisions are logged with their frames and state text, then each model answers the same recorded questions. Adding a model to the comparison costs a replay, not a new run of the sim.

**On the third attempt at a fix, build the diagnostic.**
The bin misses got two guessed fixes first (fill the far side first: 11 of 20, worse; shift the aim by a measured 5 cm: 10 of 20, worse). A 60-line script that printed the wrist, the held part and the resting part for each placement showed in one run that the aim and the reach were right and the release was throwing the part.

**A small clean test set hides a common failure.**
"Clean episodes 8 of 8" and "4 of 4" were true and missed that any episode with three parts for one bin failed. Twenty fresh seeds found it at once.

**Keep the bench scripts.**
Every number above came from a script kept in `output/bench`, so it could be re-run after the move to Windows and compared like for like.

- **A published benchmark rate carries a by-episode bootstrap interval, computed by the scoring code.** Decisions inside one episode share the parts, the bins and the camera, so the episode is the unit; with 60 aborts and 48 "no" answers in 974 decisions, a difference between two arms is often inside the noise. `benchmark.py score` prints the interval beside every rate since Oct 14; a number without one does not go in the paper, the slides or the README. (Oct 14 critique of the 974-decision benchmark.)

- **When a result lands, re-read every slide and every section, not only the ones that show numbers.** On Oct 14 the four results slides got the test03 numbers while the "ladder" slide still showed Gemma as rung 2 and the "on-device" slide still said 12 GB and "[Jetson __]", both contradicted by the new headline ladder (the 27B is rung 2, about 20 GB) and by Steven's Oct 14 answer (AGX Orin 64 GB). A deck is one argument; a stale method slide undoes a fresh results slide. Same for PAPER.md: the Setup and Limits sections change when the results do.


- **A resource figure on a slide is read with nvidia-smi at the setting the benchmark used, in the same session, never derived from a log's component breakdown at another setting.** On Oct 14 the on-device slide went out as v9 saying 16.7 GB for the 27B (13.1 + 0.96 + 2.6 from the 16k serve log) and "fits in a third of the robot's memory"; the measurement an hour later read 19.1 GB at 4k, and the two rungs do not fit one 3090 together. When the measurement needs a yes, the slide says "not measured" until it has one.
