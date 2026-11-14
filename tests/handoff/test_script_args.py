"""The run scripts refuse flag combinations that would silently do nothing."""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["demo", "run_sort"])
def test_close_with_the_force_grip_is_refused_not_ignored(name, capsys):
    """--close sets the fixed grasp's closure; grasp_force never reads it (run_trials.py refuses it the same way)."""
    parse = script(name).parse_args
    with pytest.raises(SystemExit) as stop:
        parse(["--grip", "force", "--close", "0.75"])
    assert stop.value.code == 2 and "--grip force" in capsys.readouterr().err
    assert parse(["--grip", "fixed", "--close", "0.75"]).close == 0.75
    assert parse(["--grip", "force"]).close is None
    with pytest.raises(SystemExit) as stop:
        parse(["--grip", "touch", "--close", "0.85"])
    assert stop.value.code == 2 and "--grip touch" in capsys.readouterr().err
    assert parse(["--grip", "touch"]).grip == "touch"


def test_the_demo_defaults_to_the_touch_grip_and_a_bare_close_means_fixed():
    """dup's pick (Oct 27): the demo closes with the touch grip; --close alone still sets the fixed grasp, not an error."""
    parse = script("demo").parse_args
    assert parse([]).grip == "touch"
    args = parse(["--close", "0.85"])
    assert (args.grip, args.close) == ("fixed", 0.85)


def test_the_benchmark_runner_keeps_the_fixed_grip_so_prereg_commands_rerun_as_recorded():
    assert script("run_sort").parse_args([]).grip == "fixed"


def test_speech_reader_is_its_own_rung_on_the_ladders_server():
    demo = script("demo")
    frozen = demo.speech_reader("frozen")
    assert (frozen.name, frozen.url, frozen.data_url, frozen.expect) == ("d3-27b-vision", demo.D3_URL, True, ("d3-27b",))
    assert demo.speech_reader("frozen") is not frozen  # the ladder wraps its own rung to log; speech must not pass through that
    assert demo.speech_reader("local").name == "decider-2b"
    assert demo.speech_reader("oracle") is None
