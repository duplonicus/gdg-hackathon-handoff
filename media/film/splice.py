"""Join the Kaggle film: the intro, then film.html's render with each footage card replaced by its clip (Nov 9).

The clips are the recorded takes in output/film/ (media/film/FOOTAGE.md). Each 'F' segment of film.js's FILM list is
cut out of out/film.mp4 and the matching clip, trimmed to the same length, goes in its place, so every slide keeps its
timing. Labels ("sped up") are drawn in the film's handwriting font and overlaid. Silent after the intro: the voice
is on hold (dup, Nov 9). Run from WSL with the motion venv (pillow):

    ~/dev/motion/.venv/bin/python media/film/splice.py            # -> media/film/out/film_rough.mp4

Prints the segment table it used, so the cut can be checked against film.js.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILM_OUT = ROOT / "output" / "film"
# the intro's render: media/intro/out/ (gitignored); FILM_INTRO overrides (the approved Nov 6 draft was rendered in the
# capsule worktree, .claude/worktrees/capsules/media/intro/out/intro_draft_sound.mp4)
INTRO = Path(os.environ.get("FILM_INTRO", ROOT / "media" / "intro" / "out" / "intro_sound.mp4"))  # make_sound.py out/intro.mp4
FPS = 30
FF = "ffmpeg"

# label -> list of (clip file, start s, end s, speed); pieces play back to back to fill the footage slot
CLIPS = {
    "A": [("p1b/front.mp4", 53.0, 69.0, 1.0)],                     # part_12, one whole cycle: reach, grasp, carry, rework bin, arm home (P1-A)
    "B": [("p1b/p1b_front.mp4", 38.37, 47.37, 1.0)],               # finish part_10: d3 0.58, the page, the answer (P1-B)
    "C": [("p2a/closeup.mp4", 21.0, 29.5, 1.0)],                   # a leaker pinched off the parchment and dropped in the bin; ends with the hand out (dup, Nov 11: after 30 s a capsule moves on its own)
    "D": [("p1d/p1d_front.mp4", 20.0, 28.0, 1.0),                  # d3 is down: the questions climb to the page (P1-D)
          ("p1d/p1d_front.mp4", 28.0, 41.0, 5.0),                  # the 120 s with no answer, sped up
          ("p1d/p1d_front.mp4", 41.0, 43.4, 1.0)],                 # the safe answer, the run carries on
}
# Times are in the composed clips, where compose_operator_shot.py holds each page question 2.5 s and each press 0.8 s;
# they come from output/film/<take>/composed_index.jsonl (P1-B question at 39.37 s; P1-D's unanswered question
# appears at 26.43 s and goes at 41.2 s).
# label -> list of (text, start s within the slot, end s)
LABELS = {
    "D": [("120 s with no answer, sped up", 8.0, 10.6), ("No answer: the safe answer, defective", 10.6, 11.7)],
}


def film_segments() -> list[dict]:
    """The FILM list from film.js, with start times, the same way film.js lays it out."""
    src = (HERE / "film.js").read_text(encoding="utf-8")
    move = float(re.search(r"const MOVE = ([0-9.]+);", src).group(1))
    body = src[src.index("const FILM = [") + len("const FILM = ["): src.index("];", src.index("const FILM = ["))]
    segs, t = [], 0.0
    for m in re.finditer(r"\[([^\[\]]+)\]", body):
        parts = [p.strip().strip("'") for p in m.group(1).split(",")]
        d = move if parts[-1] == "MOVE" else float(parts[-1])
        segs.append({"k": parts[0], "label": parts[1] if parts[0] == "F" else None, "t0": round(t, 4), "d": d})
        t += d
    return segs


def label_png(text: str, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:  # the film's handwriting font (FreeType reads WOFF2 when built with brotli)
        font = ImageFont.truetype(str(HERE / "assets" / "patrick.woff2"), 54)
    except OSError:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46)
    w = int(font.getlength(text)) + 60
    img = Image.new("RGBA", (w, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, 89], radius=18, fill=(251, 250, 245, 235), outline=(43, 43, 43, 255), width=3)
    d.text((30, 12), text, font=font, fill=(192, 57, 43, 255))
    img.save(path)


def run(args: list[str]) -> None:
    subprocess.run([FF, "-v", "error", "-y", *args], check=True)


def main() -> int:
    out = HERE / "out"
    work = out / "splice"
    work.mkdir(parents=True, exist_ok=True)
    film = out / "film.mp4"
    segs = film_segments()
    print(json.dumps([{k: v for k, v in s.items() if v is not None} for s in segs]))
    # 1. each footage slot as its own clip, exactly its slot's length
    for s in segs:
        if s["k"] != "F":
            continue
        lab = s["label"]
        pieces = []
        for i, (f, a, b, speed) in enumerate(CLIPS[lab]):
            p = work / f"{lab}_{i}.mp4"
            run(["-ss", f"{a}", "-to", f"{b}", "-i", str(FILM_OUT / f), "-an", "-vf", f"setpts=PTS/{speed},fps={FPS},scale=1920:1080", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(p)])
            pieces.append(p)
        lst = work / f"{lab}.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in pieces), encoding="utf-8")
        joined = work / f"{lab}_joined.mp4"
        run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
        slot = work / f"{lab}_slot.mp4"
        vf = f"fps={FPS},tpad=stop_mode=clone:stop_duration=2,trim=duration={s['d']},setpts=PTS-STARTPTS"
        inputs, chain = ["-i", str(joined)], f"[0:v]{vf}[v0]"
        for j, (text, t0, t1) in enumerate(LABELS.get(lab, [])):
            png = work / f"{lab}_label{j}.png"
            label_png(text, png)
            inputs += ["-i", str(png)]
            chain += f";[v{j}][{j + 1}:v]overlay=x=(W-w)/2:y=60:enable='between(t,{t0},{t1})'[v{j + 1}]"
        last = f"[v{len(LABELS.get(lab, []))}]"
        # -r: an overlay graph (or tpad/trim) otherwise comes out at 25 fps, which a stream-copied join plays wrong
        run([*inputs, "-filter_complex", chain, "-map", last, "-r", str(FPS), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(slot)])
    # 2. the slides between the slots, cut from the render, and the slots, in order
    parts, k, i = [], 0, 0
    while i < len(segs):
        if segs[i]["k"] == "F":
            parts.append(work / f"{segs[i]['label']}_slot.mp4"); i += 1; continue
        j = i
        while j < len(segs) and segs[j]["k"] != "F":
            j += 1
        a, b = segs[i]["t0"], segs[j - 1]["t0"] + segs[j - 1]["d"]
        p = work / f"slides_{k:02d}.mp4"; k += 1
        run(["-ss", f"{a}", "-to", f"{b}", "-i", str(film), "-an", "-vf", f"fps={FPS},scale=1920:1080", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(p)])
        parts.append(p)
        i = j
    # 3. the intro (scaled to 1080p, its sound kept), then the rest with silence under it
    if not INTRO.exists():
        raise SystemExit(f"no intro render at {INTRO}: render media/intro/intro.html, or set FILM_INTRO")
    intro = work / "intro_1080.mp4"
    run(["-i", str(INTRO), "-vf", f"fps={FPS},scale=1920:1080:flags=lanczos", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "1", str(intro)])
    lst = work / "body.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    body = work / "body.mp4"
    for p in parts:  # a stream-copied join needs every piece at the same frame rate
        rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
        if rate != f"{FPS}/1":
            raise SystemExit(f"{p.name} is {rate} fps, not {FPS}")
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(body)])
    body_a = work / "body_audio.mp4"
    run(["-i", str(body), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-shortest", "-c:v", "copy", "-c:a", "aac", str(body_a)])
    final = out / "film_rough.mp4"
    lst = work / "all.txt"
    lst.write_text(f"file '{intro.name}'\nfile '{body_a.name}'\n", encoding="utf-8")
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(final)])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(final)], capture_output=True, text=True).stdout.strip()
    print(f"written {final} ({float(dur):.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
