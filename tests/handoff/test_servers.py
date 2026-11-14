"""demo.py starts a Decider that is down and waits for it, instead of telling the person to start it."""

import pytest

import handoff.servers as servers
from handoff.servers import decider_start_command, ensure_decider


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_a_running_decider_is_left_alone():
    starts = []
    assert ensure_decider(healthy=lambda: True, start=lambda: starts.append(1), say=lambda _: None) == "running"
    assert starts == []


def test_a_down_decider_is_started_once_and_waited_for():
    clock, starts, said = Clock(), [], []
    answers = iter([False, False, False, True])
    state = ensure_decider(healthy=lambda: next(answers, True), start=lambda: starts.append(1), sleep=clock.sleep, clock=clock, say=said.append)
    assert state == "started" and starts == [1]
    assert any("starting" in line for line in said) and any("up after" in line for line in said)


def test_a_decider_that_never_answers_raises_with_where_its_log_is():
    clock = Clock()
    where = "serve.log" if servers.os.name == "nt" else "journalctl --user -u handoff-decider"
    with pytest.raises(RuntimeError, match=where):
        ensure_decider(timeout_s=30, healthy=lambda: False, start=lambda: None, sleep=clock.sleep, clock=clock, say=lambda _: None)


def test_start_command_is_the_native_offline_server_on_windows_and_a_systemd_unit_on_linux(monkeypatch):
    """On Windows never wsl.exe: the WSL venv's cache went empty after the move to D: and a start began a download (Oct 21)."""
    monkeypatch.delenv("HANDOFF_DECIDER_START", raising=False)
    windows = decider_start_command("http://127.0.0.1:8765", platform="nt")
    other_port = decider_start_command("http://127.0.0.1:8766", platform="nt")
    linux = decider_start_command("http://127.0.0.1:8766", platform="posix")
    assert windows == ["cmd.exe", "/c", servers.DECIDER_WINDOWS_SERVE] and "wsl.exe" not in windows
    assert other_port[-2:] == ["--port", "8766"]
    assert linux[0] == "systemd-run" and "--port 8766 --vision" in linux[-1] and servers.DECIDER_CHECKPOINT in linux[-1]


def test_start_command_can_be_replaced_from_the_environment(monkeypatch):
    monkeypatch.setenv("HANDOFF_DECIDER_START", "D:\\decider\\serve.ps1")
    assert decider_start_command(platform="nt") == ["cmd.exe", "/c", "D:\\decider\\serve.ps1"]
    assert decider_start_command(platform="posix") == ["bash", "-lc", "D:\\decider\\serve.ps1"]


def test_commit_gate_refuses_a_load_past_the_cap_and_allows_one_under_it():
    servers.commit_gate(27.0, 80.0)  # 107 GB: under 110
    servers.commit_gate(27.0, None)  # commit unreadable: no gate
    with pytest.raises(RuntimeError, match="110 GB cap"):
        servers.commit_gate(27.0, 85.0)


def test_ensure_d3_leaves_a_running_server_alone_and_gates_a_fresh_start_on_commit():
    starts = []
    assert servers.ensure_d3(healthy=lambda: True, start=lambda: starts.append(1), commit=lambda: 90.0, gpu_free=lambda: 24000) == "running"
    assert starts == []  # already up: nothing started (only the demo's own build is gated: 90 + 8 is under 110)
    with pytest.raises(RuntimeError, match="cap"):
        servers.ensure_d3(healthy=lambda: True, start=lambda: starts.append(1), commit=lambda: 105.0, gpu_free=lambda: 24000)
    with pytest.raises(RuntimeError, match="cap"):  # 80 + 27 for d3 + 8 for the demo: past 110
        servers.ensure_d3(healthy=lambda: False, start=lambda: starts.append(1), commit=lambda: 80.0, gpu_free=lambda: 24000, say=lambda _: None)
    assert starts == []  # refused before starting anything


def test_ensure_d3_refuses_when_commit_cannot_be_read_on_windows(monkeypatch):
    monkeypatch.setattr(servers.os, "name", "nt")
    with pytest.raises(RuntimeError, match="could not read Windows commit"):
        servers.ensure_d3(healthy=lambda: False, start=lambda: None, commit=lambda: None, gpu_free=lambda: 24000, say=lambda _: None)


def test_ensure_d3_refuses_a_gpu_without_room(monkeypatch):
    monkeypatch.setattr(servers.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="GB free"):
        servers.ensure_d3(healthy=lambda: False, start=lambda: None, commit=lambda: 50.0, gpu_free=lambda: 10_000, say=lambda _: None)


def test_ensure_d3_starts_and_waits_under_its_own_name():
    clock, starts, said = Clock(), [], []
    answers = iter([False, False, True])
    state = servers.ensure_d3(healthy=lambda: next(answers, True), start=lambda: starts.append(1), commit=lambda: 50.0, gpu_free=lambda: 24000,
                              sleep=clock.sleep, clock=clock, say=said.append)
    assert state == "started" and starts == [1]
    assert said[0].startswith("d3 27B at http://127.0.0.1:8774 is not running")


class Exited:
    pid = 1

    def poll(self):
        return 3


def test_a_server_that_exits_during_its_load_fails_at_once_not_after_the_timeout():
    clock = Clock()
    with pytest.raises(RuntimeError, match="exited during its load"):
        servers.ensure_d3(healthy=lambda: False, start=Exited, commit=lambda: 50.0, gpu_free=lambda: 24000,
                          sleep=clock.sleep, clock=clock, say=lambda _: None)
    assert clock.now < 10


def stop_scene(monkeypatch, owner):
    """A server on Windows that answers until something stops it; records what was killed and which unit stopped."""
    state = {"up": True, "killed": [], "units": []}
    monkeypatch.setattr(servers.os, "name", "nt")
    monkeypatch.setattr(servers.time, "sleep", lambda _: None)
    monkeypatch.setattr(servers, "listener", lambda port, tries=6: owner)
    monkeypatch.setattr(servers, "server_by_port_flag", lambda port: None)  # no process found by its --port flag either

    def kill(pid):
        state["killed"].append(pid)
        state["up"] = False

    def run(argv, **kw):
        state["units"].append(argv[-1])
        state["up"] = False

    monkeypatch.setattr(servers, "kill_tree", kill)
    monkeypatch.setattr(servers.subprocess, "run", run)
    return state


def test_stop_server_kills_a_windows_server_by_its_port(monkeypatch):
    state = stop_scene(monkeypatch, (4242, "llama-server"))
    assert servers.stop_server(8774, lambda: state["up"], "d3 27B", say=lambda _: None, wait_s=0.0)
    assert state["killed"] == [4242] and state["units"] == []


def test_stop_server_never_kills_the_wsl_relay_and_halts_the_wsl_unit(monkeypatch):
    """Under NAT localhost forwarding a WSL server's port belongs to wslrelay.exe; killing it cuts every WSL port."""
    state = stop_scene(monkeypatch, (999, "wslrelay"))
    assert servers.stop_server(8765, lambda: state["up"], "the Decider 2B", say=lambda _: None, wait_s=0.0, wsl_unit=servers.DECIDER_UNIT)
    assert state["killed"] == [] and state["units"] == [servers.DECIDER_UNIT]


def test_halting_d3_leaves_the_decider_unit_alone(monkeypatch):
    state = stop_scene(monkeypatch, None)
    monkeypatch.setattr(servers.DeciderRung, "healthy", lambda self: state["up"])
    assert not servers.stop_d3(say=lambda _: None)
    assert state["units"] == []


def test_listening_pid_reads_netstat():
    from handoff.servers import listening_pid

    out = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    127.0.0.1:8774         0.0.0.0:0              LISTENING       47100
  TCP    127.0.0.1:8774         127.0.0.1:59657        TIME_WAIT       0
  TCP    127.0.0.1:18774        0.0.0.0:0              LISTENING       999
  TCP    [::1]:8765             [::]:0                 LISTENING       4321
"""
    assert listening_pid(out, 8774) == 47100
    assert listening_pid(out, 8765) == 4321
    assert listening_pid(out, 877) is None  # 18774 and 8774 do not match 877
    assert listening_pid(out, 59657) is None  # a TIME_WAIT remote side is not a listener


def test_stop_server_falls_back_to_the_port_flag_when_the_tcp_table_misses_it(monkeypatch):
    import os

    import pytest

    from handoff import servers

    if os.name != "nt":
        pytest.skip("the Windows kill path")
    up = {"v": True}
    killed = []
    monkeypatch.setattr(servers, "listener", lambda port, tries=6: None)
    monkeypatch.setattr(servers, "server_by_port_flag", lambda port: 4242)
    monkeypatch.setattr(servers, "kill_tree", lambda pid: (killed.append(pid), up.update(v=False)))
    said = []
    assert servers.stop_server(8774, lambda: up["v"], "d3 27B", said.append, wait_s=1.0)
    assert killed == [4242]
    assert any("--port 8774" in s for s in said)
