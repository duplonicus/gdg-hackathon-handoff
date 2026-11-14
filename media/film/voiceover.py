"""Lay dup's voice-over on the film (Nov 10): his takes cut from one recording, each placed on its scene.

The recording (output/film/vo/hackathon.m4a, a copy of his "hackathon.m4a") holds every line in film order, with
outtakes: retakes, a stray "if the", a message to Claude. Whisper large-v3 gave word timings (vo_words.json); the cut
list below names each kept piece by its source seconds; snap() then moves every cut off his breaths and onto whole
words (measured levels: a line starts 0.1 s before its first spoken window, ends after the word's tail has died). Each line is placed at a scene: an absolute time in the intro, or a film.js FILM segment plus an
offset, so the slide holds in film.js and these placements stay in step. The intro's sound effects play only up to
the end of the cloud scene (dup: "get rid of the sound effects after the cloud dropping models scene"). Loudness is
compressed and normalized toward -14 LUFS in 32-bit float. Writes media/film/out/film_vo.mp4 (the rough cut's video, this audio).

    ~/dev/motion/.venv/bin/python media/film/voiceover.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VO = ROOT / "output" / "film" / "vo"
SR = 48000
INTRO_LEN = 87.033
SFX_END = 42.83  # the intro's Phase 1 title card (TT in intro.js): the cloud scene is over
FF = "ffmpeg"

# name -> (pieces as (source start, source end) seconds, anchor, offset s). Anchor "intro" = absolute intro time;
# otherwise ("M", to-slide) or ("F", label): the start of that FILM segment.
# Offsets were re-set when snap() stopped keeping the lead-in breath (Nov 10), so each word plays when it did before.
CUTS = {
    "thought":   ([(1.52, 9.54)], "intro", 17.65),            # Maybe decision models would be well suited for robotics...
    "ohboy":     ([(12.66, 15.84)], "intro", 37.96),          # Oh boy, what have I gotten myself into.
    "phase1":    ([(330.88, 333.62)], "intro", 43.67),        # Phase one, painted part inspection. (his retake at the end)
    "finishq":   ([(25.90, 29.16)], "intro", 51.08),          # Is the finish on this part good or defective?
    "vlmvdm":    ([(32.94, 43.28)], "intro", 55.15),          # A VLM reads the picture and writes words. A VDM ... a probability.
    "ladder":    ([(52.96, 67.44)], "intro", 70.33),          # How can we get the best result? ... more reliable.
    "cell":      ([(69.10, 73.90)], ("M", 3), 1.31),          # In simulation, the MH3 sorted defective and good parts
    "questions": ([(73.90, 85.88)], ("M", 4), 3.05),          # the model answered four questions ... is the sort done
    "tree":      ([(87.85, 90.96)], ("M", 5), 0.85),          # behavior tree was responsible for the rest of the steps (starts after "done" ends)
    "frame":     ([(96.58, 101.02), (101.85, 110.94), (112.42, 115.50)], ("M", 6), 1.44),  # reads the frame; If the probability ... 80%; nobody answers
    "pageview":  ([(117.82, 121.96)], ("F", "B"), 1.73),      # Here we have a view of the operator page when the robot isn't sure.
    "results":   ([(123.94, 130.00), (131.90, 136.10), (137.90, 144.98)], ("M", 11), 2.03),  # Over 300 ... D3 | decided alone ... of the time, | handed off ... 2.6% (his two 1.8 s pauses cut, Nov 11)
    "bridge":    ([(146.86, 154.20)], ("M", 12), 2.21),       # but which kind of model is more trustworthy, a VLM or a VDM
    "phase2":    ([(154.20, 168.50), (169.15, 171.16)], ("M", 13), 2.67),  # phase 2, a more realistic Windsor based job ... removed | from good soft gel capsules (his hesitation after "removed" cut, Nov 11)
    "leakers":   ([(186.90, 191.54)], ("F", "C"), 1.33),       # his retake, whole: "Leakers are detected by a pool of liquid on a sheet of parchment paper" (it starts with its own "Leakers"; the first take's kept "Leakers" made a stutter, Nov 11)
    "capsuleq":  ([(193.94, 197.20)], ("M", 14), 1.34),       # Is the capsule in the middle of this picture a leaker?
    "qwend3":    ([(198.52, 212.28)], ("M", 14), 5.09),       # Qwen wrote good, 95% sure. d3 98% leaker.
    "test05":    ([(234.44, 245.02)], ("M", 15), 1.98),       # Over 467 leaker pictures ... each wrong one time. (second take)
    "qwen":      ([(246.84, 249.46), (281.74, 286.36)], ("M", 15), 13.26),  # Qwen 3.8 27B, which D3 was based from, wrong over a quarter
    "down":      ([(291.16, 295.46)], ("M", 9), 1.33),        # If the model goes down, every question goes to the operator page.
    "nobody":    ([(295.46, 299.62)], ("F", "D"), 7.41),      # If nobody answers, the safe option is chosen.
    "vram":      ([(306.64, 310.22)], ("M", 10), 1.56),       # And all of this fits in 24 gigabytes of VRAM. (second take)
    "close":     ([(313.18, 320.10)], ("M", 16), 2.88),       # Logic for what must never fail ... a person when it isn't sure.
}
GAP_BETWEEN_PIECES = 0.35  # silence put between the pieces of one line
GAP_IN_SENTENCE = {"results": [0.0, 0.25], "phase2": [0.0]}  # silence added at each join of these lines (the cuts keep ~0.15 s): 0 mid-sentence, 0.25 at a comma


def film_segments() -> list[dict]:
    src = (HERE / "film.js").read_text(encoding="utf-8")
    move = float(re.search(r"const MOVE = ([0-9.]+);", src).group(1))
    body = src[src.index("const FILM = [") + len("const FILM = ["): src.index("];", src.index("const FILM = ["))]
    segs, t = [], 0.0
    for m in re.finditer(r"\[([^\[\]]+)\]", body):
        p = [x.strip().strip("'") for x in m.group(1).split(",")]
        d = move if p[-1] == "MOVE" else float(p[-1])
        segs.append({"k": p[0], "key": (p[0], int(p[2])) if p[0] == "M" else (p[0], p[1] if p[0] == "F" else int(p[1])), "t0": t, "d": d})
        t += d
    return segs


def level(x, sr: int, t: float, win: float = 0.02) -> float:
    """dB RMS of the 20 ms window starting at t."""
    import numpy as np

    seg = x[max(0, int(t * sr)): max(0, int((t + win) * sr))]
    return float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)) if len(seg) else -180.0


TAIL = -62.0    # dB: a 20 ms window below this is past a word's tail (room tone about -90; tails of "into", "steps" sit at -45 to -55)
SPEECH = -45.0  # dB: a window above this is a spoken word; his inhales before a line sit at -55 to -65


def snap(a: float, b: float, x, sr: int) -> tuple[float, float]:
    """Cut points moved off his breaths and onto whole words (Nov 10, dup: "gotten myself in" for "into", inhales at
    the starts). Start: past any sound still running at a (the previous word's tail), then 0.1 s before the first
    spoken window within 2.5 s of a: Whisper's word starts sit up to 0.9 s early, with the inhale in between. End: on to the
    first 80 ms below TAIL within 0.8 s of Whisper's word end, so a quiet tail is kept. Speech running on past either
    limit (a cut inside a sentence) falls back to the quietest 20 ms nearby."""
    def quietest(lo: float, hi: float) -> float:
        ts = [lo + i * 0.01 for i in range(int((hi - lo) / 0.01))]
        return min(ts, key=lambda t: level(x, sr, t))

    def quiet_run(t: float) -> bool:
        return all(level(x, sr, t + k * 0.02) < TAIL for k in range(4))

    # the onset search starts at a (or past the sound running there), never before it: a word inside the lookback
    # would be a fragment of the line before; only the 40 ms before a are known quiet, so the lead-in stops there
    start = None
    lo, floor = a, a - 0.04
    if level(x, sr, a - 0.04, 0.04) > TAIL:
        lo = floor = next((a + i * 0.01 for i in range(60) if quiet_run(a + i * 0.01)), None)
    if lo is not None:
        on = next((lo + i * 0.01 for i in range(250) if level(x, sr, lo + i * 0.01) > SPEECH), None)
        if on is not None:
            start = max(floor, on - 0.10)
    if start is None:
        start = quietest(a - 0.3, a + 0.3)
    # a dip shorter than 80 ms is a stop consonant inside the word ("par-ts", "ste-ps"), not the end of it
    run = next((b + i * 0.01 for i in range(80) if quiet_run(b + i * 0.01)), None)
    end = run + 0.05 if run is not None else quietest(b - 0.05, b + 0.6)
    if end - start < 0.2:
        raise SystemExit(f"cut ({a}, {b}) snapped to ({start:.2f}, {end:.2f}): too short, check the cut list")
    return start, end


def load_16k():
    import wave

    import numpy as np

    with wave.open(str(VO / "vo16.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768, w.getframerate()


def main() -> int:
    segs = film_segments()
    start_of = {s["key"]: s["t0"] for s in segs}
    x, sr = load_16k()
    plan, prev_end = [], 0.0
    for name, (pieces, anchor, off) in CUTS.items():
        at = off if anchor == "intro" else INTRO_LEN + start_of[anchor] + off
        snapped = [snap(a, b, x, sr) for a, b in pieces]
        gaps = GAP_IN_SENTENCE.get(name, [GAP_BETWEEN_PIECES] * (len(snapped) - 1))
        assert len(gaps) == len(snapped) - 1, f"{name}: one gap per join"
        dur = sum(b - a for a, b in snapped) + sum(gaps)
        plan.append({"name": name, "at": round(at, 2), "dur": round(dur, 2), "gaps": gaps, "pieces": [(round(a, 2), round(b, 2)) for a, b in snapped]})
    plan.sort(key=lambda p: p["at"])
    total = INTRO_LEN + segs[-1]["t0"] + segs[-1]["d"]
    problems = []
    for p, q in zip(plan, plan[1:]):
        if p["at"] + p["dur"] > q["at"] - 0.2:
            problems.append(f"{p['name']} ({p['at']}+{p['dur']}) runs into {q['name']} at {q['at']}")
    if plan[-1]["at"] + plan[-1]["dur"] > total:
        problems.append(f"{plan[-1]['name']} runs past the end ({total:.1f} s)")
    for p in plan:
        print(f"  {p['at']:7.2f}  {p['dur']:5.2f}  {p['name']}")
    (VO / "vo_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    if problems:
        print("OVERLAPS:\n  " + "\n  ".join(problems))
        return 1
    # the voice track, built by placing samples at exact indices (an ffmpeg adelay + amix graph dropped the first
    # line's 17.2 s lead-in and slid every line early)
    import wave

    import numpy as np

    work = VO / "build"
    work.mkdir(exist_ok=True)
    with wave.open(str(VO / "vo48.wav")) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        src = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    track = np.zeros(int(round(total * SR)) + 1, dtype=np.float32)
    fade = int(0.02 * SR)
    for p in plan:
        pos = int(round(p["at"] * SR))
        for j, (a, b) in enumerate(p["pieces"]):
            piece = src[int(round(a * SR)): int(round(b * SR))].copy()
            n = min(fade, len(piece) // 2)
            piece[:n] *= np.linspace(0, 1, n, dtype=np.float32)
            piece[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
            track[pos: pos + len(piece)] += piece[: max(0, len(track) - pos)]
            pos += len(piece) + int((p["gaps"][j] if j < len(p["gaps"]) else 0) * SR)
    voice = work / "voice.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
    # the intro's effects up to the end of the cloud scene, ducked under the voice there
    sfx = ROOT / "media" / "intro" / "out" / "intro_sfx.wav"
    duck = "+".join(f"between(t,{p['at']:.2f},{p['at'] + p['dur']:.2f})" for p in plan if p["at"] < SFX_END)
    mix = work / "mix.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(voice), "-i", str(sfx), "-filter_complex",
                    f"[1:a]aresample={SR},pan=mono|c0=c0,atrim=0:{SFX_END},afade=t=out:st={SFX_END - 0.4}:d=0.4,"
                    f"volume='if({duck},0.4,1)':eval=frame,apad=whole_dur={total:.3f}[s];[0:a][s]amix=inputs=2:normalize=0[m]",
                    "-map", "[m]", "-ar", str(SR), "-ac", "1", str(mix)], check=True)
    # loudness: the voice has sharp transients (peaks about 28 dB over its average), so a firm compressor first, then
    # loudnorm (dynamic) to -14 LUFS, all in 32-bit float: 16-bit intermediates clipped at 0 dBFS and stalled at -17.5
    norm = work / "mix_norm.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(mix), "-af",
                    "aformat=sample_fmts=flt,highpass=f=80,acompressor=threshold=0.02:ratio=6:attack=2:release=100:makeup=4,"
                    f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample={SR}", "-c:a", "pcm_f32le", str(norm)], check=True)
    meas = subprocess.run([FF, "-hide_banner", "-i", str(norm), "-af", "ebur128=framelog=quiet:peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = {"I": re.findall(r"I:\s+(-?[0-9.]+) LUFS", meas)[-1], "peak": re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", meas)[-1]}
    rough = HERE / "out" / "film_rough.mp4"
    vid = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(rough)], capture_output=True, text=True).stdout)
    if abs(vid - total) > 0.1:  # film.js changed since the render: lines would sit on the wrong slides
        raise SystemExit(f"film_rough.mp4 is {vid:.2f} s but film.js adds up to {total:.2f} s: re-render film.html and re-run splice.py first")
    final = HERE / "out" / "film_vo.mp4"
    subprocess.run([FF, "-v", "error", "-y", "-i", str(HERE / "out" / "film_rough.mp4"), "-i", str(norm), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    print(f"written {final} ({total:.1f} s); loudness {m['I']} LUFS, peak {m['peak']} dBFS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
