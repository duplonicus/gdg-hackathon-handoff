#!/usr/bin/env python3
"""Synthesize the intro's sound effects and mux them onto a render. Nothing downloaded: every sound is made here.

    ~/dev/motion/.venv/bin/python make_sound.py [video.mp4]   -> out/intro_sfx.wav (+ out/<video>_sound.mp4)

The cue times come from the page itself (window.SFX: one entry per keystroke, the Enter, the whoosh of the dive
through the "g", each footstep, a pencil scratch for every drawing, and a bonk and a clatter for each falling decision
model), so the sounds stay on their moments if the timeline changes. Placeholder level only: peak
normalised to -3 dBFS, no loudnorm, because dup's voice-over will set the mix.
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import subprocess
import sys
import threading
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from playwright.async_api import async_playwright

DIR = Path(__file__).resolve().parent
OUT = DIR / "out"
SR = 48000
PORT = 8793


async def cues() -> tuple[list[dict], float]:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIR))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.add_init_script("window.__render = true")
        await pg.goto(f"http://127.0.0.1:{PORT}/intro.html")
        await pg.wait_for_function("window.ready === true", timeout=60000)
        sfx, T = await pg.evaluate("[window.SFX, window.T]")
        await b.close()
    srv.shutdown()
    return sfx, float(T)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(t / attack, 1.0) * np.exp(-np.maximum(t - attack, 0) / decay)


def key(rng: np.random.Generator, big: float = 1.0) -> np.ndarray:
    n = int(.09 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    click = np.diff(noise, prepend=0) * env(n, .0008, .006 * big) * .5          # the plastic click (high-passed noise)
    f = rng.uniform(170, 260) / big
    thock = np.sin(2 * np.pi * f * t) * env(n, .002, .022 * big) * .9            # the key bottoming out
    return (click + thock) * rng.uniform(.75, 1.0)


def whoosh(rng: np.random.Generator, dur: float) -> np.ndarray:
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    # a low-pass whose cutoff rises through the dive (one-pole filter, coefficient swept), level rising then cut
    a = np.linspace(.01, .35, n) ** 1.4
    y = 0.0
    for i in range(n):
        y += a[i] * (noise[i] - y)
        out[i] = y
    u = np.linspace(0, 1, n)
    return out * (u ** 2.2) * np.minimum((1 - u) / .06, 1) * 2.2


def step(rng: np.random.Generator) -> np.ndarray:
    """A soft footstep: a low thud and a little scuff."""
    n = int(.12 * SR)
    t = np.arange(n) / SR
    thud = np.sin(2 * np.pi * rng.uniform(70, 95) * t) * env(n, .003, .03)
    scuff = lowpass(rng.standard_normal(n), .08) * env(n, .002, .02) * 1.5
    return (thud + scuff) * rng.uniform(.8, 1.0)


def lowpass(x: np.ndarray, a: float) -> np.ndarray:
    y, out = 0.0, np.empty_like(x)
    for i in range(len(x)):
        y += a * (x[i] - y)
        out[i] = y
    return out


def scratch(rng: np.random.Generator, dur: float) -> np.ndarray:
    """A pencil on paper: band-limited noise in quick strokes (chka-chka-chka), about 9 strokes a second."""
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    hp = noise - lowpass(noise, .25)          # drop the rumble
    band = lowpass(hp, .55)                   # and the hiss
    t = np.arange(n) / SR
    rate = rng.uniform(7.5, 10.5)
    strokes = np.clip(np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)), 0, None) ** 1.5
    fade = np.minimum(np.minimum(t / .03, 1), np.minimum((dur - t) / .08, 1))
    return band * (.35 + .65 * strokes) * fade


def roll(rng: np.random.Generator, dur: float) -> np.ndarray:
    """Wheels on a factory floor: a low rumble that slows to a stop (the robot eases in)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rumble = lowpass(lowpass(rng.standard_normal(n), .02), .05) * 6
    ticks = np.clip(np.sin(2 * np.pi * np.cumsum(np.linspace(14, 2, n)) / SR), 0, None) ** 8 * .25
    return (rumble + ticks * lowpass(rng.standard_normal(n), .3)) * np.minimum(t / .1, 1) * (1 - t / dur) ** 1.5


def bonk(rng: np.random.Generator) -> np.ndarray:
    """A small wooden block off a head: a pitched knock that drops a little."""
    n = int(.25 * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(520, 640)
    ph = 2 * np.pi * np.cumsum(f0 * (1 - .25 * np.minimum(t / .08, 1))) / SR
    tone = np.sin(ph) * env(n, .001, .05) + .4 * np.sin(2.7 * ph) * env(n, .001, .02)
    click = np.diff(rng.standard_normal(n), prepend=0) * env(n, .0005, .003) * .4
    return tone + click


def clatter(rng: np.random.Generator) -> np.ndarray:
    """A box landing on the pile: two or three quick knocks."""
    out = np.zeros(int(.3 * SR))
    for k in range(rng.integers(2, 4)):
        s = bonk(rng)[: int(.12 * SR)] * rng.uniform(.3, .6)
        i = int(rng.uniform(0, .12) * SR) + k * int(.03 * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[: j - i]
    return out


def main() -> None:
    sfx, T = asyncio.run(cues())
    rng = np.random.default_rng(438)
    track = np.zeros(int((T + .5) * SR))

    def put(sig: np.ndarray, t: float, gain: float) -> None:
        i = int(t * SR)
        j = min(len(track), i + len(sig))
        track[i:j] += sig[: j - i] * gain

    for c in sfx:
        if c["kind"] == "key":
            put(key(rng), c["t"], .22)
        elif c["kind"] == "space":
            put(key(rng, 1.35), c["t"], .26)
        elif c["kind"] == "enter":
            put(key(rng, 1.7), c["t"], .34)
        elif c["kind"] == "whoosh":
            put(whoosh(rng, c["dur"]), c["t"], .12)
        elif c["kind"] == "step":
            put(step(rng), c["t"], .22)
        elif c["kind"] == "scratch":
            put(scratch(rng, c["dur"]), c["t"], .05)
        elif c["kind"] == "roll":
            put(roll(rng, c["dur"]), c["t"], .3)
        elif c["kind"] == "bonk":
            put(bonk(rng), c["t"], .2)
        elif c["kind"] == "clatter":
            put(clatter(rng), c["t"], .1)
    track *= 10 ** (-3 / 20) / max(1e-9, np.abs(track).max())
    OUT.mkdir(exist_ok=True)
    wav = OUT / "intro_sfx.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((track * 32767).astype("<i2").tobytes())
    print("wrote", wav, f"{len(sfx)} cues, {T:.2f} s")
    if len(sys.argv) > 1:
        vid = Path(sys.argv[1]).resolve()
        dst = vid.with_name(vid.stem + "_sound.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(vid), "-i", str(wav),
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", str(dst)], check=True)
        print("wrote", dst)


if __name__ == "__main__":
    main()
