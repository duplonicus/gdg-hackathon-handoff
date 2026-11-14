"""Start the model server the demo needs when it is down, instead of telling the person to.

The Decider (``strands-decider serve``) is the one process this project owns.

- On Windows (the project's home since Oct 11, and where GPU work runs) it is the native server
  ``D:\\Models\\decision-servers\\decider\\serve.cmd``: its own venv, weights from the hard-link mirror
  ``D:\\Models\\hf-win`` with ``HF_HUB_OFFLINE=1``, so a start can never download weights. It is started detached
  (it outlives the demo) with its output in ``serve.log`` beside it.
- On Linux it runs as a transient systemd user unit named ``handoff-decider``, output in the journal.

Until Oct 21 the Windows start went through ``wsl.exe`` to the old WSL venv. After the model cache moved to D: that
venv's default cache was empty, and a demo start began downloading the 2B weights unasked (stopped at 225 MB).

Ollama is a Docker container or ``D:\\Ollama\\start.ps1``, not started from here.

``HANDOFF_DECIDER_START`` replaces the whole start command (a shell line) for a Decider served some other way,
for instance from a Windows venv once the transformers models move to native Windows.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from typing import Any, Callable
from urllib.parse import urlparse

from .rungs import DECIDER_URL, DeciderRung

DECIDER_CHECKPOINT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
DECIDER_SERVE = "~/dev/decider-server/.venv/bin/strands-decider"
DECIDER_UNIT = "handoff-decider"
DECIDER_WINDOWS_SERVE = r"D:\Models\decision-servers\decider\serve.cmd"
DECIDER_WINDOWS_LOG = r"D:\Models\decision-servers\decider\serve.log"
# Measured Oct 13 on this PC: the 2B checkpoint answers /health 16 s after the unit starts (model on the WSL vdisk on C:).
DECIDER_LOAD_TIMEOUT_S = 300.0


def decider_log_command() -> str:
    """What to run to read the server's output when it did not come up."""
    if os.name == "nt":
        return f'type "{DECIDER_WINDOWS_LOG}"'
    return f"journalctl --user -u {DECIDER_UNIT} -n 50 --no-pager"


def decider_start_command(url: str = DECIDER_URL, platform: str = os.name) -> list[str]:
    """The argv that starts the Decider for ``url`` (its port) as the ``handoff-decider`` user unit."""
    override = os.environ.get("HANDOFF_DECIDER_START")
    if override:
        return ["cmd.exe", "/c", override] if platform == "nt" else ["bash", "-lc", override]
    port = urlparse(url).port or 8765
    if platform == "nt":
        extra = [] if port == 8765 else ["--port", str(port)]  # serve.cmd says 8765; a later --port wins
        return ["cmd.exe", "/c", DECIDER_WINDOWS_SERVE, *extra]
    serve = f"{DECIDER_SERVE} serve {DECIDER_CHECKPOINT} --port {port} --vision"
    return ["systemd-run", "--user", f"--unit={DECIDER_UNIT}", "--collect", "bash", "-lc", serve]


def start_detached(argv: list[str], log_path: str) -> subprocess.Popen:
    """Start a Windows server that outlives the caller, its output in ``log_path``.

    A hidden console, not DETACHED_PROCESS: with no console at all, cmd.exe's console child (llama-server.exe,
    strands-decider.exe) got a new visible console of its own, its output went there, and the log stayed empty
    (the Decider's serve.log was 0 bytes on Oct 21; closing that window would have killed the server)."""
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)


def log_tail(path: str, n: int = 15) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:]).strip()
    except OSError:
        return ""


def kill_tree(pid: int) -> None:
    """Stop a Windows process and everything it started (cmd.exe and the server under it)."""
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)


def start_decider(url: str = DECIDER_URL) -> None:
    """Launch the server. On Windows: detached, output to serve.log. On Linux: the unit; a unit that already exists
    (still loading from an earlier start) is not an error."""
    if os.name == "nt" and not os.environ.get("HANDOFF_DECIDER_START"):
        start_detached(decider_start_command(url), DECIDER_WINDOWS_LOG)  # the Decider runs on after the demo
        return
    done = subprocess.run(decider_start_command(url), capture_output=True, text=True, check=False)
    if done.returncode != 0 and "already exists" not in done.stderr:
        raise RuntimeError(f"could not start the Decider: {shlex.join(decider_start_command(url))}\n{done.stderr.strip()}")


def ensure_decider(
    url: str = DECIDER_URL,
    timeout_s: float = DECIDER_LOAD_TIMEOUT_S,
    *,
    healthy: Callable[[], bool] | None = None,
    start: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
    label: str = "the Decider",
    load_hint: str = "the model takes about 20 s to load",
    log_hint: str | None = None,
    alive: Callable[[], int | None] | None = None,
    on_timeout: Callable[[], None] | None = None,
) -> str:
    """Return ``"running"`` if the server already answers, ``"started"`` once a fresh start does.

    Raises RuntimeError, with the command that shows its log, when it has not answered after ``timeout_s``.
    """
    healthy = healthy or DeciderRung("decider-2b", url).healthy
    start = start or (lambda: start_decider(url))
    if healthy():
        return "running"
    say(f"{label} at {url} is not running: starting it ({load_hint})...")
    start()
    began = clock()
    last_word = began
    while clock() - began < timeout_s:
        sleep(2.0)
        if healthy():
            say(f"{label} is up after {clock() - began:.0f} s")
            return "started"
        code = alive() if alive is not None else None
        if code is not None:  # the server exited: say so now, not after the whole timeout
            raise RuntimeError(f"{label} exited during its load (exit code {code}). Its log: {log_hint or decider_log_command()}")
        if clock() - last_word >= 15.0:
            say(f"  still loading ({clock() - began:.0f} s)")
            last_word = clock()
    if on_timeout is not None:
        on_timeout()  # leave nothing half-loaded holding the GPU
    raise RuntimeError(f"{label} at {url} did not answer /health within {timeout_s:.0f} s of starting. Its log: {log_hint or decider_log_command()}")


# ----- d3 27B, the frozen ladder's model (docs/PREREG.md) -----

D3_URL = "http://127.0.0.1:8774"
D3_SERVE = r"D:\Toolkits\build\serve_d3-27b.cmd"  # the PREREG's llama-server flags; outside the repo
D3_LOG = r"D:\Toolkits\build\serve_d3-27b.log"
D3_COMMIT_GB = 27.0  # a d3 27B GPU load added 26 to 27 GB of Windows commit (Oct 18 and 19 logs)
D3_GPU_MIB = 19_000  # d3 27B took about 17.7 to 19.2 GB of GPU memory in those loads
DEMO_COMMIT_GB = 8.0  # the demo's Genesis build, viewer and panel: an estimate, not measured yet
COMMIT_CAP_GB = 110.0  # dup's cap since Oct 20 (limit 128)


def _powershell(cmd: str, timeout: float = 30.0) -> str:
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=timeout, check=True).stdout


def windows_commit_gb() -> float | None:
    """Windows commit in use (GB): RAM plus page file promised to programs. None when it cannot be read."""
    try:
        out = _powershell("$o=Get-CimInstance Win32_OperatingSystem; ($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB")
        return float(out.strip().replace(",", "."))
    except Exception:  # noqa: BLE001 - not on Windows, PowerShell unavailable, or too slow
        return None


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return None


def commit_gate(adds_gb: float, used_gb: float | None, cap_gb: float = COMMIT_CAP_GB, *, required: bool = False) -> None:
    """Refuse a load that would take Windows commit past the cap (the Oct 13 freeze was commit, not RAM).

    ``required``: refuse when commit cannot be read at all (on Windows a slow read is exactly when it matters)."""
    if used_gb is None:
        if required:
            raise RuntimeError("could not read Windows commit, so not loading d3 27B (it adds about 27 GB). Check memory, then run again.")
        return
    if used_gb + adds_gb > cap_gb:
        raise RuntimeError(f"commit is {used_gb:.1f} GB; adding about {adds_gb:.0f} GB would pass the {cap_gb:.0f} GB cap. Free memory first.")


def ensure_d3(
    url: str = D3_URL,
    timeout_s: float = 600.0,
    *,
    healthy: Callable[[], bool] | None = None,
    start: Callable[[], subprocess.Popen | None] | None = None,
    commit: Callable[[], float | None] = windows_commit_gb,
    gpu_free: Callable[[], int | None] = gpu_free_mib,
    **kw: Any,
) -> str:
    """Start the d3 27B llama-server (serve_d3-27b.cmd, hidden, output in its log) unless it already answers.

    Gates first: commit for d3 plus the demo's own build under the cap, and the GPU free enough to hold d3. Fails
    fast if the server exits during the load, and stops it if the load times out, so nothing is left half-loaded."""
    healthy = healthy or DeciderRung("d3-27b-vision", url).healthy
    if healthy():
        commit_gate(DEMO_COMMIT_GB, commit(), required=os.name == "nt")
        return "running"
    commit_gate(D3_COMMIT_GB + DEMO_COMMIT_GB, commit(), required=os.name == "nt")
    free = gpu_free()
    for _ in range(15):  # memory a model just unloaded can take a few seconds to show as free
        if free is None or free >= D3_GPU_MIB:
            break
        time.sleep(1.0)
        free = gpu_free()
    if free is not None and free < D3_GPU_MIB:
        raise RuntimeError(f"the GPU has {free / 1024:.1f} GB free; d3 27B needs about {D3_GPU_MIB / 1024:.0f} GB. Close what holds it (a game, ComfyUI, another model) and run again.")
    proc: dict[str, subprocess.Popen | None] = {"p": None}

    def launch() -> None:
        proc["p"] = (start or (lambda: start_detached(["cmd.exe", "/c", D3_SERVE], D3_LOG)))()

    def alive() -> int | None:
        return proc["p"].poll() if proc["p"] is not None else None

    def on_timeout() -> None:
        if proc["p"] is not None and os.name == "nt":
            kill_tree(proc["p"].pid)

    try:
        return ensure_decider(url, timeout_s, healthy=healthy, start=launch, label="d3 27B", load_hint="a 15 GB model onto the GPU",
                              log_hint=f'type "{D3_LOG}"', alive=alive, on_timeout=on_timeout, **kw)
    except RuntimeError as error:
        tail = log_tail(D3_LOG)
        raise RuntimeError(f"{error}\nlast lines of the log:\n{tail}" if tail else str(error)) from None


# WSL's own processes, which own a WSL server's port on Windows under NAT localhost forwarding. Killing one cuts every
# WSL port, not the server. This PC's .wslconfig runs mirrored mode (read Oct 21), where a WSL server shows no Windows
# listener at all, so there the stop falls through to the WSL unit.
WSL_PROCESSES = frozenset({"wslrelay", "wslservice", "wslhost", "vmmem", "vmmemwsl"})


def listening_pid(netstat_out: str, port: int) -> int | None:
    """The PID LISTENING on local ``port`` in ``netstat -ano`` output (IPv4 or IPv6), or None."""
    for line in netstat_out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING" and parts[1].rsplit(":", 1)[-1] == str(port) and parts[4].isdigit():
            return int(parts[4])
    return None


def listener(port: int, tries: int = 6) -> tuple[int, str] | None:
    """The Windows process listening on a local port, as (pid, process name without .exe), or None.

    Windows' TCP table read back incomplete on Nov 7 (netstat and Get-NetTCPConnection alike listed a different ~10
    of the listeners on each call, so d3's port showed in 3 of 10 reads), so a miss is retried before it counts."""
    for k in range(tries):
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=30, check=True).stdout
            pid = listening_pid(out, port)
            if pid:
                return pid, _powershell(f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).ProcessName").strip()
        except Exception:  # noqa: BLE001
            pass
        if k + 1 < tries:
            time.sleep(0.3)
    return None


def server_by_port_flag(port: int) -> int | None:
    """The PID of a Windows process started with ``--port <port>`` on its command line (llama-server, our Python
    servers): the fallback when the TCP table misses a listener that /health shows is up."""
    try:
        out = _powershell(f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '--port\\s+{port}(\\s|$)' }} | "
                          "Select-Object -First 1 -ExpandProperty ProcessId")
        return int(out.strip()) if out.strip().isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def stop_server(port: int, healthy: Callable[[], bool], label: str, say: Callable[[str], None] = print, wait_s: float = 20.0, wsl_unit: str | None = None) -> bool:
    """Stop whatever serves ``port``, found by the port (not by command-line text), and confirm it went down.

    Windows: the listening process's tree, unless that process is WSL's own (a WSL-side server behind localhost
    forwarding: killing wslrelay would cut every WSL port). If /health still answers, ``wsl_unit``, the systemd user
    unit that serves it on the WSL side, is stopped too (None: there is no such unit). Returns True when the port no
    longer answers."""
    if not healthy():
        return True
    if os.name == "nt":
        owner = listener(port)
        if owner and owner[1].lower() not in WSL_PROCESSES:
            kill_tree(owner[0])
        elif owner is None and (pid := server_by_port_flag(port)):
            say(f"{label}: port {port} missing from the TCP table; stopping the process started with --port {port} (pid {pid})")
            kill_tree(pid)
    t0 = time.monotonic()
    while healthy() and time.monotonic() - t0 < wait_s:
        time.sleep(1.0)
    if healthy() and wsl_unit:
        subprocess.run((["wsl.exe", "-e"] if os.name == "nt" else []) + ["systemctl", "--user", "stop", wsl_unit], capture_output=True, check=False)
        time.sleep(3.0)
    down = not healthy()
    say(f"stopped {label} on port {port}" if down else f"could NOT stop {label} on port {port}: it still answers")
    return down


def stop_decider(say: Callable[[str], None] = print) -> bool:
    """Stop the Decider 2B (about 4.8 GB of GPU memory) so d3 27B fits."""
    return stop_server(urlparse(DECIDER_URL).port or 8765, DeciderRung("decider-2b").healthy, "the Decider 2B", say, wsl_unit=DECIDER_UNIT)


def stop_d3(say: Callable[[str], None] = print) -> bool:
    """Stop d3 27B (about 18 GB of GPU memory) before a ladder that loads other models."""
    return stop_server(urlparse(D3_URL).port or 8774, DeciderRung("d3-27b-vision", D3_URL).healthy, "d3 27B", say)
