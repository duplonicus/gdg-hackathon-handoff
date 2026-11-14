"""The demo's text in the Genesis viewer: laid out so it never touches Genesis's own help text, and fail-soft."""

import importlib.util
import re
from pathlib import Path
from typing import ClassVar

import pytest

from handoff import viewer_text as vt

WINDOWS = [(400, 300), (720, 540), (1024, 768), (1440, 900), (2560, 1440)]
MODELS = "d3 27B: UP   Decider: DOWN   Gemma: Ollama is up; gemma4:e4b loads on its first question"
LINES = [
    "12:00:01  finish part_03: d3-27b defective 0.91 => defective (by d3-27b) and a long tail of words to cut " * 2,
    "12:00:02  event: bump_good_wall_ym — retry at z 0.31 … then a very long explanation of the retry " * 2,
]
# what [i] expands to: the collapsed line plus one line per keybind (about 20 in the demo's viewer)
EXPANDED = [vt.DEFAULT_HELP] + [f"{'[' + k:>7}]: some key action name" for k in "wasdqezxcvbnmrtfgyhu"]


def genesis_dir():
    spec = importlib.util.find_spec("genesis")  # finds the package without importing it (it pulls in torch)
    return Path(spec.origin).parent if spec and spec.origin else None


def real_measure():
    """Widths from the real Genesis fonts, the way font.render_string adds them up (advance >> 6)."""
    freetype = pytest.importorskip("freetype")
    root = genesis_dir()
    if root is None or not (root / "ext/pyrender/fonts/OpenSans-Regular.ttf").exists():
        pytest.skip("Genesis fonts not installed")
    faces, advances = {}, {}

    def measure(text, size, font):
        total = 0
        for c in text:
            key = (font, size, c)
            if key not in advances:
                if (font, size) not in faces:
                    face = freetype.Face(str(root / "ext/pyrender/fonts" / f"{font}.ttf"))
                    face.set_pixel_sizes(0, int(size))
                    faces[(font, size)] = face
                face = faces[(font, size)]
                face.load_char(c)
                advances[key] = face.glyph.advance.x >> 6
            total += advances[key]
        return total

    return measure


def wide_measure(text, size, font):
    """Pessimistic stand-in: every proportional glyph a full em wide; the monospace help font is half an em."""
    return len(text) * size * (0.5 if font == vt.HELP_FONT else 1.0)


def boxes(placed, measure):
    return [vt.text_box(measure(t, s, vt.FONT), x, y, a, s) for t, x, y, a, s in placed]


def check_clear(w, h, placed, measure, help_lines):
    hb = vt.help_box(h, help_lines, measure)
    got = boxes(placed, measure)
    for i, box in enumerate(got):
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h, (w, h, placed[i])
        if hb is not None:
            assert vt.apart(box, hb), (w, h, placed[i], hb)
        for other in got[i + 1 :]:
            assert vt.apart(box, other, 0, 0), (w, h, placed[i])
    return hb


@pytest.mark.parametrize("w,h", WINDOWS)
@pytest.mark.parametrize("which", ["real", "wide"])
def test_no_box_touches_the_collapsed_help_text_at_any_width(w, h, which):
    measure = real_measure() if which == "real" else wide_measure
    placed = vt.layout(w, h, MODELS, LINES, measure)
    hb = check_clear(w, h, placed, measure, [vt.DEFAULT_HELP])
    by_align = {a: [p for p in placed if p[3] == a] for a in ("top_right", "bottom_left")}
    # the models line is always shown, and on the help text's row it starts at least GAP_X right of it
    assert len(by_align["top_right"]) == 1
    models_box = boxes(by_align["top_right"], measure)[0]
    if models_box[3] == h - vt.TEXT_PADDING:
        assert models_box[0] >= hb[2] + vt.GAP_X
    # both log lines are shown, newest on the bottom row, each cut by pixels to the left half of the window
    assert len(by_align["bottom_left"]) == 2
    for (text, *_), source in zip(by_align["bottom_left"], reversed(LINES), strict=True):
        assert vt.ascii_text(source).startswith(text.removesuffix("..."))
    assert by_align["bottom_left"][0][2] == vt.TEXT_PADDING and by_align["bottom_left"][1][2] > vt.TEXT_PADDING
    for text, *_, size in by_align["bottom_left"]:
        assert measure(text, size, vt.FONT) <= w / 2 - 40 and text.endswith("...")


def test_the_real_help_text_box_is_where_genesis_draws_it():
    measure = real_measure()
    hb = vt.help_box(540, [vt.DEFAULT_HELP], measure)
    assert hb[0] == 20 and hb[2] == pytest.approx(449, abs=2) and hb[3] == 520


def test_the_old_top_center_caption_did_overlap_at_the_default_window():
    """The bug this module fixes: demo.py put a 90-character log line at TOP_CENTER (16 px)."""
    measure = real_measure()
    w, h = 720, 540
    text = LINES[0][:90]
    width = measure(text, 16, vt.FONT)
    old = (w / 2 - width / 2, h - 20 - 16 * 1.3, w / 2 + width / 2, h - 20)
    assert not vt.apart(old, vt.help_box(h, [vt.DEFAULT_HELP], measure))


@pytest.mark.parametrize("w,h", WINDOWS)
@pytest.mark.parametrize("which", ["real", "wide"])
def test_nothing_touches_the_expanded_key_list_either(w, h, which):
    measure = real_measure() if which == "real" else wide_measure
    placed = vt.layout(w, h, MODELS, LINES, measure, help_lines=EXPANDED)
    check_clear(w, h, placed, measure, EXPANDED)


def test_bottom_lines_give_way_to_a_key_list_that_reaches_them_and_come_back_when_it_folds():
    measure = real_measure()
    expanded = vt.layout(720, 540, MODELS, LINES, measure, help_lines=EXPANDED)
    assert not [p for p in expanded if p[3] == "bottom_left"]
    collapsed = vt.layout(720, 540, MODELS, LINES, measure)
    assert len([p for p in collapsed if p[3] == "bottom_left"]) == 2


def test_no_help_text_means_the_models_line_gets_the_whole_row():
    placed = vt.layout(720, 540, "Models: NOT USED", [], wide_measure, help_lines=[])
    assert placed == [("Models: NOT USED", 700.0, 520.0, "top_right", vt.MODELS_SIZE)]


def test_truncate_to_px_cuts_by_pixels_to_the_longest_prefix_that_fits():
    measure = lambda t: 10 * len(t) + 5 * t.count("W")  # a wide W shows it is pixels, not characters
    assert vt.truncate_to_px("short", measure, 100) == "short"
    cut = vt.truncate_to_px("WWWWWWWWWWWWWW", measure, 100)
    assert cut == "WWWW..." and measure(cut) <= 100
    cut = vt.truncate_to_px("abcdefghijklmnop", measure, 100)
    assert cut == "abcdefg..." and measure(cut) <= 100
    assert vt.truncate_to_px("abcdefghijklmnop", measure, 101) == "abcdefg..."
    assert vt.truncate_to_px("abcdefghijklmnop", measure, 29) == ""  # not even the ellipsis fits
    assert vt.truncate_to_px("abc   defghijklmnop", measure, 90) == "abc..."  # no space left before the ellipsis


def test_text_is_kept_to_the_ascii_the_genesis_font_holds():
    """font.render_string looks every character up in a 0-127 map: anything else raised KeyError in the GL thread."""
    assert vt.ascii_text("a — b … c→d café\nnext") == "a - b ... c->d caf? next"
    for text, *_ in vt.layout(1440, 900, MODELS, LINES, wide_measure):
        assert all(32 <= ord(c) < 127 for c in text)


def test_align_codes_match_genesis():
    root = genesis_dir()
    path = root / "ext/pyrender/constants.py" if root else None
    if path is None or not path.exists():
        pytest.skip("Genesis not installed")
    source = path.read_text(encoding="utf-8")
    for name, code in vt.ALIGN.items():
        assert re.search(rf"^\s+{name.upper()} = (\d+)", source, re.MULTILINE).group(1) == str(code)


# ----- install(): the GL side, with a stand-in viewer -----
class Glyph:
    advance = 10 << 6


class FakeFont:
    _character_map: ClassVar[dict] = {chr(i): Glyph() for i in range(128)}


class FakeFontCache:
    def get_font(self, name, size):
        return FakeFont()


class FakeRenderer:
    dpscale = 1

    def __init__(self):
        self._font_cache = FakeFontCache()
        self.drawn = []

    def render_text(self, text, x, y, **kwargs):
        self.drawn.append((text, x, y, kwargs["align"], kwargs["font_pt"]))


class FakeViewer:
    def __init__(self):
        self._renderer = FakeRenderer()
        self.viewport_size = (1440, 900)
        self._enable_help_text = True
        self._collapse_instructions = True
        self._instr_texts = ([vt.DEFAULT_HELP], ["< [i]: hide keyboard instructions"])
        self._key_instr_texts = EXPANDED
        self.help_drawn = 0

    def _render_help_text(self):
        self.help_drawn += 1


def test_install_draws_after_genesis_help_text_every_frame():
    viewer = FakeViewer()
    vt.install(viewer, lambda: ("d3 27B: UP", ["12:00:01  first", "12:00:02  second"]))
    viewer._render_help_text()
    viewer._render_help_text()
    assert viewer.help_drawn == 2
    frame = viewer._renderer.drawn[:3]
    assert frame == [
        ("d3 27B: UP", 1420.0, 880.0, vt.ALIGN["top_right"], 20),
        ("12:00:02  second", 20.0, 20.0, vt.ALIGN["bottom_left"], 16),
        ("12:00:01  first", 20.0, 44.0, vt.ALIGN["bottom_left"], 16),
    ]
    assert viewer._renderer.drawn[3:] == frame  # the second frame draws the same, from the cached layout


def test_install_follows_the_help_text_when_it_expands():
    viewer = FakeViewer()
    viewer.viewport_size = (720, 540)
    vt.install(viewer, lambda: ("d3 27B: UP", ["12:00:01  first"]))
    viewer._collapse_instructions = False
    viewer._render_help_text()
    assert [d[0] for d in viewer._renderer.drawn if d[3] == vt.ALIGN["bottom_left"]] == []


def test_a_viewer_missing_a_private_attribute_logs_once_and_draws_nothing():
    viewer = FakeViewer()
    del viewer._renderer
    said = []
    vt.install(viewer, lambda: ("models", ["line"]), log=said.append)
    viewer._render_help_text()
    viewer._render_help_text()
    assert viewer.help_drawn == 2  # Genesis's own help text keeps drawing
    assert len(said) == 1 and said[0].startswith("viewer text off")


def test_a_failing_provider_logs_once_per_kind_and_the_viewer_keeps_going():
    viewer = FakeViewer()
    said, calls = [], []

    def provider():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("status lock broke")
        return "models", ["line"]

    vt.install(viewer, provider, log=said.append)
    for _ in range(3):
        viewer._render_help_text()
    assert viewer.help_drawn == 3
    assert said == ["viewer text skipped a frame: RuntimeError: status lock broke"]
    assert [d[0] for d in viewer._renderer.drawn] == ["models", "line"]  # drawn again once the provider recovers


def test_a_viewer_without_render_help_text_is_left_alone():
    said = []
    viewer = object()
    vt.install(viewer, lambda: ("models", []), log=said.append)
    assert said == ["viewer text off: this Genesis viewer has no _render_help_text to draw after"]
