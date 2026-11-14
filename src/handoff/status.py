"""What a person watching a run needs to see: are the models up, and what has happened so far.

One RunStatus per run. A background thread asks the model servers every few seconds; the run adds a line for every
decision and event. The kit's control panel and the viewer both show it, and it is written to run.log beside the
decision log. Sensors (the tray load cells first) each get a window on the panel through ``set_sensor``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decider import Decision, RungAnswer
from .rungs import DECIDER_URL, OLLAMA_URL, _get, served_model

D3_URL = "http://127.0.0.1:8774"


def check_decider(url: str = DECIDER_URL) -> dict[str, Any]:
    try:
        health = _get(f"{url}/health", timeout=2.0)
    except Exception as exc:  # noqa: BLE001
        return {"name": "Decider", "ok": False, "detail": f"not running ({type(exc).__name__})"}
    ok = health.get("status") == "ok"
    return {"name": "Decider", "ok": ok, "detail": f"loaded: {health.get('model', 'unknown model')}" if ok else f"status {health.get('status')}"}


def check_d3(url: str = D3_URL) -> dict[str, Any]:
    """The frozen ladder's model. Port 8774 is shared with other 27B arms, so "up" also means the name says d3."""
    try:
        health = _get(f"{url}/health", timeout=2.0)
    except Exception as exc:  # noqa: BLE001
        return {"name": "d3 27B", "ok": False, "detail": f"not running ({type(exc).__name__})"}
    if health.get("status") not in ("ok", "ready"):
        return {"name": "d3 27B", "ok": False, "detail": f"status {health.get('status')}"}
    name = served_model(url)
    if "d3-27b" not in name:
        return {"name": "d3 27B", "ok": False, "detail": f"port 8774 serves another model: {name[:80]}"}
    return {"name": "d3 27B", "ok": True, "detail": "loaded: d3 27B (llama-server)"}


def check_gemma(model: str = "gemma4:e4b", url: str = OLLAMA_URL) -> dict[str, Any]:
    """Ollama keeps a model in memory only for a while after it was last asked, so "up" and "loaded" are different."""
    try:
        running = [m.get("name", "") for m in _get(f"{url}/api/ps", timeout=2.0).get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return {"name": "Gemma", "ok": False, "detail": f"Ollama not running ({type(exc).__name__})"}
    if any(name.startswith(model) for name in running):
        return {"name": "Gemma", "ok": True, "detail": f"loaded: {model}"}
    return {"name": "Gemma", "ok": True, "detail": f"Ollama is up; {model} loads on its first question"}


def rung_text(r: RungAnswer) -> str:
    """What one rung said: "ERROR", "unclear" when it gave every option the same chance (the scales when the readings
    do not settle it; "scale yes 0.50" read as a half-sure yes), else its top answer and confidence."""
    if r.error:
        return f"{r.rung} ERROR"
    if len(r.probs) > 1 and max(r.probs.values()) - min(r.probs.values()) < 1e-9:
        return f"{r.rung} unclear"
    return f"{r.rung} {r.top} {r.confidence:.2f}"


def describe(decision: Decision) -> str:
    """One line per decision: who was asked, what each said, who answered."""
    part = decision.context.get("part", "")
    climb = " -> ".join(rung_text(r) for r in decision.rungs)
    return f"{decision.question_id} {part}: {climb} => {decision.answer} (by {decision.answered_by})"


def finish_line(status: str, checks: Mapping[str, Any], decisions: Sequence[Decision]) -> str:
    """The run's last log line. The tree's "success" means the sort was verified complete (and the robot danced, when dancing is on), not
    that every part is in its right bin: a part the ladder called good when it was defective is "sorted" into the
    wrong tray. So a mis-sort is named, with the finish answer behind it when that answer was wrong."""
    n, correct = int(checks.get("n_parts", 0)), int(checks.get("parts_in_correct_bin", 0))
    if correct < n:
        why = []
        for part, where in (checks.get("mis_sorted") or {}).items():
            finish = [d for d in decisions if d.question_id == "finish" and d.context.get("part") == part]
            d = finish[-1] if finish else None
            if d is not None and d.truth is not None and d.answer != d.truth:
                why.append(f"finish {part}: answered {d.answer} by {d.answered_by}, truth {d.truth}")
            else:
                why.append(f"{part}: {where}")
        head = f"{'completed' if status == 'success' else status}, {n - correct} of {n} parts mis-sorted" + (f" ({'; '.join(why)})" if why else "")
    else:
        head = f"{status}, {correct} of {n} parts in the right bin"
    return f"run finished: {head}; bumps {checks.get('bumps', 0)}; checks {dict(checks)}"


def bump_rows(bumps: Sequence[Mapping[str, Any]], last: int = 5) -> list[dict[str, str]]:
    """The "Arm bumps" sensor window: how much higher the arm now crosses, then the newest bumps (box, height, what
    the arm did about it)."""
    extra = float(bumps[-1].get("clearance_extra", 0.0)) if bumps else 0.0
    rows = [{"label": "crossing extra", "value": f"+{extra * 100:.0f} cm", "detail": f"{len(bumps)} bump{'s' if len(bumps) != 1 else ''} so far", "state": "warn" if bumps else "ok"}]
    for b in list(bumps)[-last:][::-1]:
        retry = b.get("retry_z")
        rows.append({"label": str(b.get("box")), "value": f"wrist {float(b.get('wrist_z', 0.0)):.3f} m", "detail": f"{b.get('skill')}: " + (f"retried at {float(retry):.3f} m" if retry is not None else "let go there"), "state": "warn"})
    return rows


class RunStatus:
    def __init__(self, ladder: str, log_path: Path | None = None, max_lines: int = 200, title: str | None = None) -> None:
        self.ladder = ladder
        self.title = title or f"{ladder} ladder"
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._models: list[dict[str, Any]] = []
        self._sensors: dict[str, dict[str, Any]] = {}  # sensor windows on the panel, in the order first set
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = log_path.open("a", encoding="utf-8")
        self.check_models()

    # ----- models -----
    def check_models(self) -> list[dict[str, Any]]:
        if self.ladder == "oracle":
            models = [{"name": "Models", "ok": True, "state": "NOT USED", "detail": "oracle ladder: the simulator's truth answers every question"}]
        elif self.ladder == "frozen":
            models = [check_d3()]
        else:
            models = [check_decider(), check_gemma()]
        with self._lock:
            changed = [m for m, old in zip(models, self._models or [{}] * len(models), strict=False) if m.get("ok") != old.get("ok")]
            first = not self._models
            self._models = models
        if not first:
            for m in changed:
                self.log(f"{m['name']} is now {'UP' if m['ok'] else 'DOWN'}: {m['detail']}")
        return models

    def start(self, period_s: float = 5.0) -> None:
        def loop() -> None:
            while not self._stop.wait(period_s):
                self.check_models()

        threading.Thread(target=loop, name="run-status", daemon=True).start()

    # ----- the run's log -----
    def log(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        with self._lock:
            self._lines.append(line)
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()

    def decision(self, decision: Decision) -> None:
        self.log(describe(decision))

    # ----- sensor windows -----
    def set_sensor(self, name: str, rows: list[dict[str, str]]) -> None:
        """Replace one sensor window's rows ({"label", "value", "detail", "state": "ok" | "warn" | "idle"}) and stamp it.

        A new name adds a window after the ones already set; the panel draws each window as one tile."""
        window = {"name": name, "updated": time.strftime("%H:%M:%S"), "rows": [dict(r) for r in rows]}
        with self._lock:
            self._sensors[name] = window

    # ----- what the screens show -----
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            sensors = [{**w, "rows": [dict(r) for r in w["rows"]]} for w in self._sensors.values()]
            return {"title": f"Sorting run ({self.title})", "badges": list(self._models), "lines": list(self._lines), "sensors": sensors}

    def caption_lines(self, n: int = 2) -> tuple[str, list[str]]:
        with self._lock:
            models = "   ".join(f"{m['name']}: {m.get('state') or ('UP' if m['ok'] else 'DOWN')}" for m in self._models)
            return models, list(self._lines)[-n:]

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
