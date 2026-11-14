"""The replay chain's stop (Nov 7): a server the TCP table misses is still stopped, and "gone" waits for /health."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("replay_capsules", ROOT / "scripts" / "replay_capsules.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def scene(monkeypatch, owner, flag_pid):
    state = {"up": True, "killed": []}
    monkeypatch.setattr(rc, "listener", lambda port, tries=6: owner if state["up"] else None)
    monkeypatch.setattr(rc, "server_by_port_flag", lambda port: flag_pid)
    monkeypatch.setattr(rc, "healthy", lambda port: state["up"])
    monkeypatch.setattr(rc.time, "sleep", lambda _: None)

    def kill(pid):
        state["killed"].append(pid)
        state["up"] = False

    monkeypatch.setattr(rc, "kill_tree", kill)
    return state


def test_a_listed_server_is_killed_by_its_port(monkeypatch):
    state = scene(monkeypatch, (4242, "llama-server"), None)
    rc.stop(8774)
    assert state["killed"] == [4242]


def test_a_server_missing_from_the_table_is_found_by_its_port_flag(monkeypatch):
    state = scene(monkeypatch, None, 777)
    rc.stop(8774)
    assert state["killed"] == [777]


def test_stop_does_not_call_a_live_server_gone_on_a_table_miss(monkeypatch):
    state = scene(monkeypatch, None, None)  # nothing found to kill: it must still be waiting on /health, not return early
    calls = {"n": 0}

    def health(port):
        calls["n"] += 1
        return state["up"]

    monkeypatch.setattr(rc, "healthy", health)
    rc.stop(8774)
    assert calls["n"] >= 30  # waited the full time, since /health kept answering
