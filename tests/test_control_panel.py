"""Test control panel camera encoding and batch frame processing."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mh3_genesis.control_panel import MH3ControlPanel


def test_encode_rgb() -> None:
    """Verify RGB image encoding produces a valid JPEG payload.
    :return: None when encoded bytes deserialize into an RGB image.
    :rtype: None
    """
    # Create a synthetic 8-bit RGB test pattern.
    raw_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_rgb[:, :, 0] = 120
    raw_rgb[:, :, 1] = 200
    # Compress RGB array to JPEG image bytes.
    encoded_jpeg = MH3ControlPanel._encode_rgb(raw_rgb, format_name="jpeg", quality=75)
    # Confirm JPEG header magic bytes.
    assert encoded_jpeg.startswith(b"\xff\xd8\xff")
    image_jpeg = Image.open(io.BytesIO(encoded_jpeg))
    assert image_jpeg.format == "JPEG"
    assert image_jpeg.size == (640, 480)

    # Compress RGB array to PNG image bytes.
    encoded_png = MH3ControlPanel._encode_rgb(raw_rgb, format_name="png")
    assert encoded_png.startswith(b"\x89PNG\r\n\x1a\n")
    image_png = Image.open(io.BytesIO(encoded_png))
    assert image_png.format == "PNG"
    assert image_png.size == (640, 480)


def test_encode_depth() -> None:
    """Verify metric depth encoding produces a valid grayscale JPEG or PNG preview.
    :return: None when depth array transforms and encodes into requested image formats.
    :rtype: None
    """
    # Create a synthetic floating-point metric depth array containing non-finite and distant values.
    raw_depth = np.full((480, 640), 2.5, dtype=np.float32)
    raw_depth[0, 0] = np.nan
    raw_depth[0, 1] = np.inf
    raw_depth[0, 2] = -1.0
    raw_depth[0, 3] = 10.0
    # Compress metric depth buffer to JPEG image bytes.
    encoded_jpeg = MH3ControlPanel._encode_depth(raw_depth, near_m=0.1, far_m=5.0, format_name="jpeg", quality=85)
    # Confirm JPEG header magic bytes.
    assert encoded_jpeg.startswith(b"\xff\xd8\xff")
    image_jpeg = Image.open(io.BytesIO(encoded_jpeg))
    assert image_jpeg.format == "JPEG"
    assert image_jpeg.size == (640, 480)

    # Compress metric depth buffer to PNG image bytes.
    encoded_png = MH3ControlPanel._encode_depth(raw_depth, near_m=0.1, far_m=5.0, format_name="png")
    assert encoded_png.startswith(b"\x89PNG\r\n\x1a\n")
    image_png = Image.open(io.BytesIO(encoded_png))
    assert image_png.format == "PNG"
    assert image_png.size == (640, 480)


def test_all_camera_frames() -> None:
    """Verify all_camera_frames packages cached frame buffers into data URLs with proper MIME types.
    :return: None when frame dictionary is formatted as base64 data URLs.
    :rtype: None
    """
    import threading
    import time

    # Create an uninitialized panel instance to test frame batching in isolation.
    panel = MH3ControlPanel.__new__(MH3ControlPanel)
    panel._lock = threading.Lock()
    panel._last_camera_request_at = 0.0
    panel._camera_frames = {
        "head-left-rgb": b"\xff\xd8\xff\xe0test_jpeg",
        "head-depth": b"\x89PNG\r\n\x1a\ntest_png",
    }
    # Retrieve formatted frame payload mapping.
    payload = panel.all_camera_frames()
    # Confirm timestamp was refreshed and data URLs were correctly constructed with respective MIME types.
    assert time.monotonic() - panel._last_camera_request_at < 1.0
    assert payload["head-left-rgb"].startswith("data:image/jpeg;base64,")
    assert payload["head-depth"].startswith("data:image/png;base64,")


def test_state_with_extra_adds_the_applications_card_and_survives_a_failing_provider() -> None:
    """The page's extra card comes from a provider the application sets; a broken provider must not break the page."""
    panel = MH3ControlPanel.__new__(MH3ControlPanel)  # no simulation needed: only the merge is under test
    panel.state = lambda: {"sim_time": 1.0}
    assert "extra" not in panel.state_with_extra()

    panel.extra_state = lambda: {"title": "Run", "badges": [{"name": "Decider", "ok": True, "detail": "loaded"}], "lines": ["a"]}
    merged = panel.state_with_extra()
    assert merged["sim_time"] == 1.0 and merged["extra"]["badges"][0]["name"] == "Decider"

    def broken():
        raise RuntimeError("boom")

    panel.extra_state = broken
    assert "boom" in panel.state_with_extra()["extra"]["lines"][0]


def test_page_has_the_extra_card_and_the_script_fills_it() -> None:
    from mh3_genesis.control_panel import _html
    from mh3_genesis.dashboard_script import SCRIPT

    page = _html("token", 0.6, 1.0)
    assert 'id="extra-panel"' in page and 'id="extra-badges"' in page and 'id="extra-log"' in page
    assert "function renderExtra(" in SCRIPT and "renderExtra(state.extra);" in SCRIPT


def test_depth_tiles_say_which_camera_they_come_from() -> None:
    """Both depth tiles used to read "Left camera depth"; each now names its own camera."""
    from mh3_genesis.control_panel import _html

    page = _html("token", 0.6, 1.0)
    assert "Left camera depth" not in page
    assert "<figcaption>Head depth" in page and "<figcaption>Base depth" in page


def test_page_has_a_hidden_sensor_column_inside_the_camera_card_and_the_script_fills_it() -> None:
    from mh3_genesis.control_panel import _html
    from mh3_genesis.dashboard_script import SCRIPT
    from mh3_genesis.dashboard_style import STYLE

    page = _html("token", 0.6, 1.0)
    camera_card = page[page.index('id="camera-panel"') : page.index('id="joint-groups"')]
    assert '<aside id="sensor-panel" hidden>' in camera_card and 'id="sensor-windows"' in camera_card
    assert "function renderSensors(" in SCRIPT and "renderSensors(state.extra && state.extra.sensors);" in SCRIPT
    # The two-column layout only applies while there are sensors, and collapses under the 1000 px breakpoint.
    assert "#camera-panel.has-sensors" in STYLE
    assert "#camera-panel.has-sensors" in STYLE[STYLE.index("@media (max-width: 1000px)") :]


def _sensor_renderer_source() -> str:
    """Cut renderSensors (and the signature it keeps) out of the bundled script, to run on its own."""
    from mh3_genesis.dashboard_script import SCRIPT

    start = SCRIPT.index("let sensorSignature")
    end = SCRIPT.index("\n}\n", SCRIPT.index("function renderSensors(")) + 3
    return SCRIPT[start:end]


# A DOM just big enough for renderSensors. Assigning innerHTML throws, so the test fails if the renderer ever uses it.
_FAKE_DOM = r"""
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.className = ''; this.textContent = ''; this.hidden = false; this.title = '';
    this.classList = { set: new Set(), toggle(name, on) { on ? this.set.add(name) : this.set.delete(name); }, contains(name) { return this.set.has(name); } }; }
  append(...kids) { for (const kid of kids) { kid.parent = this; this.children.push(kid); } }
  replaceChildren(...kids) { this.children = []; this.append(...kids); }
  remove() { const i = this.parent.children.indexOf(this); if (i >= 0) this.parent.children.splice(i, 1); this.parent = null; }
  get lastElementChild() { return this.children[this.children.length - 1] || null; }
  set innerHTML(_) { throw new Error('innerHTML used'); }
}
const nodes = { 'sensor-panel': new El('aside'), 'camera-panel': new El('section'), 'sensor-windows': new El('div') };
const document = { getElementById: id => nodes[id], createElement: tag => new El(tag) };
"""

_DRIVER = r"""
const shape = () => nodes['sensor-windows'].children.map(tile => ({
  name: tile.children[0].children[0].textContent,
  updated: tile.children[0].children[1].textContent,
  rows: tile.children[1].children[0].children.map(tr => ({ cls: tr.className, cells: tr.children.map(td => td.textContent) })),
}));
const out = {};
renderSensors(undefined);
out.none = { hidden: nodes['sensor-panel'].hidden, wide: nodes['camera-panel'].classList.contains('has-sensors'), tiles: shape().length };
renderSensors([
  { name: 'Tray scales', updated: '12:00:01', rows: [
    { label: 'good tray', value: '0.21 kg', detail: '<img src=x onerror=alert(1)>', state: 'ok' },
    { label: 'reject tray', value: '0.00 kg', detail: 'empty', state: 'warn' },
    { label: 'odd', value: '?', detail: '', state: 'bogus' } ] },
  { name: 'Bumps', updated: '', rows: [] },
  { name: 'Third window', updated: '12:00:02', rows: [{ label: 'a', value: 1, detail: null, state: 'idle' }] },
]);
out.some = { hidden: nodes['sensor-panel'].hidden, wide: nodes['camera-panel'].classList.contains('has-sensors'), tiles: shape() };
renderSensors([]);
out.empty = { hidden: nodes['sensor-panel'].hidden, wide: nodes['camera-panel'].classList.contains('has-sensors') };
console.log(JSON.stringify(out));
"""


def test_sensor_renderer_draws_one_escaped_tile_per_window_and_hides_when_empty() -> None:
    """Run the real renderSensors in node against a fake DOM: any number of windows, state colours, no markup injection."""
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    program = _FAKE_DOM + _sensor_renderer_source() + _DRIVER
    result = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=60, check=True)
    out = json.loads(result.stdout)

    assert out["none"] == {"hidden": True, "wide": False, "tiles": 0}
    some = out["some"]
    assert some["hidden"] is False and some["wide"] is True
    assert [tile["name"] for tile in some["tiles"]] == ["Tray scales", "Bumps", "Third window"]
    scales = some["tiles"][0]
    assert scales["updated"] == "12:00:01"
    assert [row["cls"] for row in scales["rows"]] == ["sensor-ok", "sensor-warn", "sensor-idle"]  # an unknown state reads as idle
    assert scales["rows"][0]["cells"] == ["good tray", "0.21 kg", "<img src=x onerror=alert(1)>"]  # kept as text, never parsed
    assert some["tiles"][1]["updated"] == "no reading yet" and some["tiles"][1]["rows"][0]["cells"] == ["No rows yet."]
    assert some["tiles"][2]["rows"][0]["cells"] == ["a", "1", ""]
    assert out["empty"] == {"hidden": True, "wide": False}


_UPDATE_DRIVER = r"""
const holder = nodes['sensor-windows'];
const rows = (value) => [
  { label: 'good tray', value, detail: '1 part', state: 'ok' },
  { label: 'reject tray', value: '0.0 g', detail: '0 parts', state: 'ok' } ];
renderSensors([{ name: 'Tray load cells', updated: '12:00:01', rows: rows('51.2 g') }, { name: 'Bumps', updated: '', rows: [] }]);
const tile = holder.children[0], label = tile.children[1].children[0].children[0].children[0];
tile.mark = 'first'; label.mark = 'first';
renderSensors([{ name: 'Tray load cells', updated: '12:00:02', rows: rows('49.8 g') }, { name: 'Bumps', updated: '', rows: [] }]);
const again = holder.children[0], cells = r => r.children.map(td => td.textContent);
const out = { sameTile: again.mark === 'first', sameLabel: again.children[1].children[0].children[0].children[0].mark === 'first',
  updated: again.children[0].children[1].textContent, first: cells(again.children[1].children[0].children[0]) };
// the empty window gets rows, the first window loses one row, then a window goes away
renderSensors([{ name: 'Tray load cells', updated: '12:00:03', rows: rows('49.8 g').slice(0, 1) },
  { name: 'Bumps', updated: '12:00:03', rows: [{ label: 'bump_good_wall_ym', value: '1', detail: 'wrist_z 0.91', state: 'warn' }] }]);
out.grown = holder.children.map(t => t.children[1].children[0].children.map(tr => [tr.className, cells(tr)]));
renderSensors([{ name: 'Bumps', updated: '12:00:04', rows: [] }]);
out.shrunk = holder.children.map(t => [t.children[0].children[0].textContent, t.children[1].children[0].children.map(tr => [tr.className, cells(tr), tr.children[0].colSpan])]);
console.log(JSON.stringify(out));
"""


def test_sensor_renderer_updates_tiles_in_place_when_only_the_readings_change() -> None:
    """The load cells change value on nearly every 250 ms poll; rebuilding the tiles each time would wipe a text
    selection and the hover text of a cut-off label. The tile and an unchanged cell must be the same elements after."""
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    program = _FAKE_DOM + _sensor_renderer_source() + _UPDATE_DRIVER
    result = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=60, check=True)
    out = json.loads(result.stdout)

    assert out["sameTile"] is True and out["sameLabel"] is True
    assert out["updated"] == "12:00:02" and out["first"] == ["good tray", "49.8 g", "1 part"]
    assert out["grown"] == [
        [["sensor-ok", ["good tray", "49.8 g", "1 part"]]],
        [["sensor-warn", ["bump_good_wall_ym", "1", "wrist_z 0.91"]]],
    ]
    assert out["shrunk"] == [["Bumps", [["sensor-idle", ["No rows yet."], 3]]]]


def test_arm_sliders_say_what_each_joint_does_and_keep_the_code_name() -> None:
    """The sliders read left_1 to left_7; each now has the measured job (scripts/probe_joint_axes.py) beside its name."""
    from mh3_genesis.control_panel import joint_label
    from mh3_genesis.dashboard_script import SCRIPT

    assert [joint_label(f"left_{i}")[0] for i in range(1, 8)] == ["Shoulder pitch", "Shoulder roll", "Shoulder yaw", "Elbow", "Forearm twist", "Wrist yaw", "Wrist pitch"]
    assert joint_label("right_4") == joint_label("left_4") and joint_label("slider")[0] == "Torso lift"
    assert joint_label("left_index")[0] == "Index finger" and joint_label("head_yaw")[0] == "Head turn"
    assert joint_label("left_thumb_1")[0] == "Thumb swing" and joint_label("right_thumb_2")[0] == "Thumb close"
    assert joint_label("wheel_9") == ("wheel_9", "")  # a joint with no label keeps its own name
    assert "joint.label" in SCRIPT and "<small>${joint.name}</small>" in SCRIPT
