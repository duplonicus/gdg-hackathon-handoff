"""Replay every model arm on the capsule logs, one model on the GPU at a time, unattended (Nov 4, dup: run overnight,
load a model only when the VRAM is free and it will not spill into system RAM).

For each arm in order: check the GPU has room for it, start its server hidden, wait for it to answer, check at least
``MIN_FREE_AFTER_MIB`` is still free (an overflow into shared system memory would slow everything and could freeze the
PC), replay it on every log (``benchmark.py replay``, which skips what is already answered), then stop the server and
confirm the memory came back. A step that fails is logged and skipped; the next arm still runs. Progress goes to
``output/capsules/replay_chain.log``.

    .\\run.ps1 scripts/replay_capsules.py --logs output/capsules/sel01,output/capsules/sel01/blocks/n16 --arms d3-27b-vision,qwen-27b-vision
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.servers import commit_gate, gpu_free_mib, kill_tree, listener, server_by_port_flag, start_detached, windows_commit_gb  # noqa: E402

MIN_FREE_AFTER_MIB = 1000  # left free once a model is loaded: below this, the next allocation can spill to system RAM
QWEN_CMD = r"D:\Toolkits\build\serve_qwen-27b.cmd"
D3_CMD = r"D:\Toolkits\build\serve_d3-27b.cmd"
INTERN_CMD = [r"D:\Models\decision-servers\intern\.venv\Scripts\python.exe", str(ROOT / "scripts" / "intern_server.py"), "--model", r"D:\Models\decision-servers\intern\model", "--port", "8767"]
DECIDER_CMD = ["cmd.exe", "/c", r"D:\Models\decision-servers\decider\serve.cmd"]

# arm -> (start command, port, GPU memory it needs in MiB, Windows commit it adds in GB, log file). A 27B load added 26 to
# 27 GB of commit (servers.D3_COMMIT_GB); the smaller ones are estimates, generous on purpose.
SERVERS = {
    "d3-27b-vision": (["cmd.exe", "/c", D3_CMD], 8774, 19_500, 27.0, r"D:\Toolkits\build\serve_d3-27b.log"),
    "qwen-27b-vision": (["cmd.exe", "/c", QWEN_CMD], 8774, 19_000, 27.0, r"D:\Toolkits\build\serve_qwen-27b.log"),
    "qwen-27b-self-vision": (["cmd.exe", "/c", QWEN_CMD], 8774, 19_000, 27.0, r"D:\Toolkits\build\serve_qwen-27b.log"),
    "intern-4b-vision": (INTERN_CMD, 8767, 12_000, 20.0, str(ROOT / "output" / "capsules" / "intern_server.log")),
    "decider-2b-vision": (DECIDER_CMD, 8765, 8_000, 15.0, r"D:\Models\decision-servers\decider\serve.log"),
}


def say(line: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    text = f"{stamp} {line}"
    print(text, flush=True)
    with (ROOT / "output" / "capsules" / "replay_chain.log").open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def healthy(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            body = json.loads(r.read() or b"{}")
            return str(body.get("status", "ok")).lower() in ("ok", "ready")
    except Exception:  # noqa: BLE001
        return False


def stop(port: int) -> None:
    """Stop what serves ``port`` and wait until nothing does. The TCP table read back incomplete on Nov 7, so a server
    it misses is found by its --port flag, and "gone" means both the table and /health say so."""
    held = listener(port)
    pid = held[0] if held else (server_by_port_flag(port) if healthy(port) else None)
    if pid:
        kill_tree(pid)
    for _ in range(30):
        if not healthy(port) and not listener(port, tries=2):
            return
        time.sleep(1)


def run_arm(arm: str, logs: list[Path]) -> None:
    argv, port, need, commit_gb, log_path = SERVERS[arm]
    if listener(port) or healthy(port):
        say(f"{arm}: port {port} is already taken; stopping what holds it first")
        stop(port)
    free = gpu_free_mib()
    if free is None or free < need + MIN_FREE_AFTER_MIB:
        say(f"{arm}: SKIPPED, the GPU has {free} MiB free and it needs about {need} + {MIN_FREE_AFTER_MIB}")
        return
    try:  # Windows commit too: the Oct 13 freeze was commit, not VRAM (review round 6)
        commit_gate(commit_gb, windows_commit_gb(), required=True)
    except RuntimeError as error:
        say(f"{arm}: SKIPPED, {error}")
        return
    say(f"{arm}: starting ({free} MiB free)")
    proc = start_detached(argv, log_path)
    try:  # every way out of here kills what was started: a load that never opened its port included (review round 6)
        t0 = time.time()
        while not healthy(port):
            if proc.poll() is not None:
                say(f"{arm}: SKIPPED, the server exited during the load (log: {log_path})")
                return
            if time.time() - t0 > 900:
                say(f"{arm}: SKIPPED, no answer after 15 min")
                return
            time.sleep(5)
        after = gpu_free_mib()
        say(f"{arm}: ready after {time.time() - t0:.0f} s, {after} MiB free")
        if after is not None and after < MIN_FREE_AFTER_MIB:
            say(f"{arm}: SKIPPED, only {after} MiB left free: too close to spilling into system RAM")
            return
        for log in logs:
            if not (log / "decisions.jsonl").exists():
                say(f"{arm}: no decisions in {log}, skipped")
                continue
            t1 = time.time()
            done = subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark.py"), "replay", "--log", str(log), "--arm", arm], capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = (done.stdout.strip().splitlines() or [""])[-1]
            say(f"{arm}: {log.name or log}: exit {done.returncode} in {time.time() - t1:.0f} s: {tail[:160]}")
            if done.returncode != 0:
                say(f"{arm}: stderr: {done.stderr.strip()[-300:]}")
            if proc.poll() is not None or not healthy(port):
                say(f"{arm}: STOPPED, the server is gone (log: {log_path}); the rest of its logs are left for a rerun")
                return
    finally:
        stop(port)
        kill_tree(proc.pid)  # and the cmd.exe that started it
        time.sleep(3)
        say(f"{arm}: stopped, {gpu_free_mib()} MiB free")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", required=True, help="comma-separated log folders")
    parser.add_argument("--arms", required=True, help=f"comma-separated, in order; known: {', '.join(SERVERS)}")
    args = parser.parse_args()
    logs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.logs.split(",")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in SERVERS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    say(f"chain: {len(arms)} arms on {len(logs)} logs")
    for arm in arms:
        try:
            run_arm(arm, logs)
        except Exception as error:  # noqa: BLE001 - one arm failing must not stop the night
            say(f"{arm}: FAILED: {type(error).__name__}: {error}")
    say("chain done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
