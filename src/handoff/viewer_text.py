"""The demo's own text in the Genesis viewer: the model servers at top right, the newest run log lines at bottom left.

Genesis draws "> [i]: show keyboard instructions" at top left (UbuntuMono 26 px from x = 20, about 430 px wide) and
the full key list under it when [i] is pressed. Captions set through viewer_flags["caption"] are drawn before that
text and know nothing about it, so a log line placed at TOP_CENTER ran into it on any window under ~1500 px wide.
install() instead draws right after Genesis's help text, on the viewer's GL thread, at positions laid out from the
real glyph advances so no box can touch the help text at any window size.

layout() and truncate_to_px() are pure (a measure function stands in for the font) so they are tested without a GL
context; install() is the thin part that reads the viewer's private attributes and calls the renderer.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import Any

# Genesis's own values (genesis/ext/pyrender/constants.py and viewer._render_help_text)
TEXT_PADDING = 20
HELP_FONT = "UbuntuMono-Regular"
HELP_SIZE = 26
HELP_LINE_PITCH = 1.1  # renderer.render_texts steps each line down by font_pt * 1.1
DEFAULT_HELP = "> [i]: show keyboard instructions"

FONT = "OpenSans-Regular"
MODELS_SIZE = 20
LOG_SIZE = 16
LOG_LINE_PITCH = 1.5  # bottom-left lines are stacked upward by size * 1.5
MAX_LOG_LINES = 2
ELLIPSIS = "..."  # Genesis fonts hold only ASCII 0-127: a real ellipsis character would raise KeyError in the GL thread

GAP_X = 20  # horizontal clearance from the help text (and between our own boxes)
GAP_Y = 6  # vertical clearance
DESCENT = 0.3  # glyphs reach this fraction of the font size below the baseline (g, p, y)
MIN_MODELS_PX = 160  # narrower than this beside the help text and the models line moves under it instead
# genesis.ext.pyrender.constants.TextAlign values, copied so importing this module does not import genesis (torch)
ALIGN = {"bottom_left": 3, "top_right": 7}

# a few non-ASCII characters the run log may carry, spelled in ASCII; anything else becomes "?"
_ASCII = {"—": "-", "–": "-", "…": "...", "→": "->", "×": "x", "‘": "'", "’": "'", "“": '"', "”": '"'}

Measure = Callable[[str, int, str], float]  # (text, size px, font name) -> width px
Placed = tuple[str, float, float, str, int]  # (text, x, y, align, size); align is "top_right" or "bottom_left"
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in GL pixels (y up, origin bottom left)


def ascii_text(text: str) -> str:
    """Only characters the Genesis font atlas holds, on one line."""
    out = []
    for c in text:
        if c in "\r\n\t":
            out.append(" ")
        elif 32 <= ord(c) < 127:
            out.append(c)
        else:
            out.append(_ASCII.get(c, "?"))
    return "".join(out)


def truncate_to_px(text: str, measure: Callable[[str], float], max_px: float, ellipsis: str = ELLIPSIS) -> str:
    """The longest prefix of text (plus an ellipsis when cut) that measures at most max_px; "" when nothing fits."""
    if measure(text) <= max_px:
        return text
    if measure(ellipsis) > max_px:
        return ""
    # advances are never negative, so the cut width grows with the prefix length: binary search the longest one
    cuts = range(len(text))
    n = bisect.bisect_right(cuts, False, key=lambda k: measure(text[:k].rstrip() + ellipsis) > max_px) - 1
    return text[: max(n, 0)].rstrip() + ellipsis


def text_box(width: float, x: float, y: float, align: str, size: int) -> Box:
    """Where render_text puts the glyphs: TOP_* hangs them from y, BOTTOM_* sits the baseline on y (render_string)."""
    if align == "top_right":
        return (x - width, y - size * (1 + DESCENT), x, y)
    if align == "top_left":
        return (x, y - size * (1 + DESCENT), x + width, y)
    if align == "bottom_left":
        return (x, y - size * DESCENT, x + width, y + size)
    raise ValueError(f"unknown align {align!r}")


def help_box(viewport_h: float, help_lines: list[str] | tuple[str, ...], measure: Measure) -> Box | None:
    """The box Genesis's help text covers: render_texts draws each line TOP_LEFT, font_pt * 1.1 below the last."""
    if not help_lines:
        return None
    top = viewport_h - TEXT_PADDING
    width = max(measure(line, HELP_SIZE, HELP_FONT) for line in help_lines)
    bottom = top - (len(help_lines) - 1) * HELP_SIZE * HELP_LINE_PITCH - HELP_SIZE * (1 + DESCENT)
    return (TEXT_PADDING, bottom, TEXT_PADDING + width, top)


def apart(a: Box, b: Box, gap_x: float = GAP_X, gap_y: float = GAP_Y) -> bool:
    """True when the boxes are at least gap_x apart side by side or gap_y apart one above the other."""
    return a[0] >= b[2] + gap_x or b[0] >= a[2] + gap_x or a[1] >= b[3] + gap_y or b[1] >= a[3] + gap_y


def layout(
    viewport_w: float,
    viewport_h: float,
    models: str,
    lines: list[str],
    measure: Measure,
    help_lines: list[str] | tuple[str, ...] = (DEFAULT_HELP,),
) -> list[Placed]:
    """Where each of our texts goes, in priority order: the models line, then log lines newest first.

    lines is newest LAST. Every placed box stays inside the viewport, GAP_X / GAP_Y clear of the help text box and
    off the other placed boxes; one that cannot is left out (so the bottom-left lines vanish while a tall expanded
    key list covers them).
    """
    W, H = float(viewport_w), float(viewport_h)
    hb = help_box(H, help_lines, measure)
    taken: list[Box] = []  # our own boxes: they only must not overlap (stacked log lines sit closer than the gap)
    placed: list[Placed] = []

    def place(text: str, x: float, y: float, align: str, size: int, max_px: float) -> None:
        text = truncate_to_px(ascii_text(text), lambda t: measure(t, size, FONT), max_px)
        if not text.strip(" ."):
            return
        box = text_box(measure(text, size, FONT), x, y, align, size)
        inside = box[0] >= 0 and box[1] >= 0 and box[2] <= W and box[3] <= H
        clear = hb is None or apart(box, hb)
        if inside and clear and all(apart(box, other, 0, 0) for other in taken):
            taken.append(box)
            placed.append((text, x, y, align, size))

    # models: top right on the help text's row, its left edge kept GAP_X clear of the help text's right edge;
    # a window too narrow for that gets it on its own row under the help text
    if models:
        right = W - TEXT_PADDING
        help_right = hb[2] if hb is not None else 0.0
        room = right - (help_right + GAP_X)
        if room >= MIN_MODELS_PX or hb is None:
            place(models, right, H - TEXT_PADDING, "top_right", MODELS_SIZE, min(room, right - TEXT_PADDING))
        else:
            place(models, right, hb[1] - GAP_Y, "top_right", MODELS_SIZE, right - TEXT_PADDING)

    # log lines: newest on the bottom row, older stacked above, each cut to the left half of the window so they
    # leave room for Genesis's own bottom-right messages
    max_px = W / 2 - 2 * TEXT_PADDING
    for i, line in enumerate(reversed(lines[-MAX_LOG_LINES:])):
        place(line, TEXT_PADDING, TEXT_PADDING + i * LOG_SIZE * LOG_LINE_PITCH, "bottom_left", LOG_SIZE, max_px)
    return placed


# ----- the GL side -----
def _font_measure(renderer: Any) -> Measure:
    """Widths from the glyph advances render_string adds up, in viewport pixels (render_text scales by dpscale)."""
    scale = getattr(renderer, "dpscale", 1) or 1
    cache = renderer._font_cache

    def measure(text: str, size: int, font: str) -> float:
        chars = cache.get_font(font, size * scale)._character_map
        return sum(chars[c].advance >> 6 for c in text if c in chars) / scale

    return measure


def _shown_help(viewer: Any) -> list[str]:
    """The lines Genesis's _render_help_text is drawing this frame."""
    if not getattr(viewer, "_enable_help_text", True):
        return []
    return list(viewer._instr_texts[0] if viewer._collapse_instructions else viewer._key_instr_texts)


def install(
    viewer: Any, provider: Callable[[], tuple[str, list[str]]], log: Callable[[str], None] | None = None
) -> None:
    """Draw provider()'s (models_text, log_lines newest last) on the viewer after Genesis's help text, every frame.

    Runs on the viewer's GL thread and never raises there: a missing private attribute (a Genesis upgrade) is
    reported once through log and the text is switched off; any other error is reported once per kind and that
    frame is skipped.
    """
    report = log or print
    original = getattr(viewer, "_render_help_text", None)
    if not callable(original):
        report("viewer text off: this Genesis viewer has no _render_help_text to draw after")
        return
    state: dict[str, Any] = {"off": False, "seen": set(), "key": None, "placed": []}

    def draw() -> None:
        models, lines = provider()
        lines = list(lines)[-MAX_LOG_LINES:]
        try:
            renderer = viewer._renderer
            w, h = viewer.viewport_size
            help_lines = _shown_help(viewer)
            key = (w, h, models, tuple(lines), tuple(help_lines))
            if key != state["key"]:  # measure only when something changed, not every frame
                state["placed"] = layout(w, h, models, lines, _font_measure(renderer), help_lines)
                state["key"] = key
            color = getattr(viewer, "_font_color", (1.0, 1.0, 1.0, 0.85))
            draw_text = renderer.render_text
        except AttributeError as exc:
            state["off"] = True
            report(f"viewer text off: the Genesis viewer changed ({exc})")
            return
        for text, x, y, align, size in state["placed"]:
            draw_text(text, x, y, font_name=FONT, font_pt=size, color=color, scale=1.0, align=ALIGN[align])

    def render_help_text(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if not state["off"]:
            try:
                draw()
            except Exception as exc:  # noqa: BLE001 - the viewer thread must keep drawing the scene
                kind = type(exc).__name__
                if kind not in state["seen"]:
                    state["seen"].add(kind)
                    report(f"viewer text skipped a frame: {kind}: {exc}")
        return result

    viewer._render_help_text = render_help_text
