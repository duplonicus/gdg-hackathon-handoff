"""Charts from a benchmark.json: one self-contained HTML page with inline SVG, no dependencies.

    bash run.sh scripts/charts.py --log output/sort/test02 [--title "..."]

Writes ``<log>/charts.html``: right-decision rate and confidently-wrong rate per arm, the same
per question, the ladders as "share handed to a person" against "right rate", and latency.
Light and dark themes from the same tokens; a table view at the bottom for every chart.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

# Validated palette (dataviz skill reference instance): slots 1..4 light / dark, blue ramp, surfaces.
LIGHT = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100"}
DARK = {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#383835", "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500"}
BLUE_RAMP_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_RAMP_DARK = ["#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"]
QUESTIONS = ("finish", "safe_to_grasp", "in_bin", "sort_complete")
ARM_LABELS = {
    "decider-2b-vision": "Decider 2B, frame",
    "pplx-decider-artemis-vision": "pplx-decider 27B, frame",
    "decider-2b-text": "Decider 2B, text + description",
    "decider-e2b-text": "Decider E2B, text + description",
    "decider-e4b-text": "Decider E4B, text + description",
    "gemma-12b": "Gemma 4 12b",
    "gemma-e4b": "Gemma 4 e4b",
    "gemma-e4b-self": "Gemma 4 e4b, own confidence",
    "gemma-e2b": "Gemma 4 e2b",
    "gemma-e2b-self": "Gemma 4 e2b, own confidence",
}


def label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def hbar_chart(title: str, rows: list[tuple[str, float]], color_var: str, fmt: str = "{:.2f}", vmax: float = 1.0, note: str = "", intervals: dict[str, tuple[float, float]] | None = None) -> str:
    """Horizontal bars, one series, direct value labels in text ink, 4px rounded data end; optional 95% interval whiskers."""
    w, left, bar_h, gap, top = 640, 240, 18, 10, 36
    h = top + len(rows) * (bar_h + gap) + 28
    plot_w = w - left - (130 if intervals else 70)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}" class="chart">']
    out.append(f'<text x="0" y="18" class="title">{escape(title)}</text>')
    for k in range(5):
        x = left + plot_w * k / 4
        out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{h - 24}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{vmax * k / 4:g}</text>')
    for i, (name, v) in enumerate(rows):
        y = top + i * (bar_h + gap)
        bw = max(0.0, min(v / vmax, 1.0)) * plot_w
        out.append(f'<text x="{left - 10}" y="{y + bar_h - 4}" class="lab" text-anchor="end">{escape(name)}</text>')
        out.append(
            f'<path d="M{left},{y} h{max(bw - 4, 0):.1f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h-{max(bw - 4, 0):.1f} z" '
            f'fill="var({color_var})"><title>{escape(name)}: {fmt.format(v)}</title></path>'
        )
        lo_hi = (intervals or {}).get(name)
        if lo_hi:
            x0, x1 = (left + max(0.0, min(lo_hi[0] / vmax, 1.0)) * plot_w, left + max(0.0, min(lo_hi[1] / vmax, 1.0)) * plot_w)
            ym = y + bar_h / 2
            out.append(f'<line x1="{x0:.1f}" y1="{ym:.1f}" x2="{x1:.1f}" y2="{ym:.1f}" class="whisker"/>')
            out.append(f'<line x1="{x0:.1f}" y1="{ym - 5:.1f}" x2="{x0:.1f}" y2="{ym + 5:.1f}" class="whisker"/>')
            out.append(f'<line x1="{x1:.1f}" y1="{ym - 5:.1f}" x2="{x1:.1f}" y2="{ym + 5:.1f}" class="whisker"/>')
            out.append(f'<text x="{max(left + bw, x1) + 6:.1f}" y="{y + bar_h - 4}" class="val">{fmt.format(v)} <tspan class="tick">[{fmt.format(lo_hi[0])}, {fmt.format(lo_hi[1])}]</tspan></text>')
        else:
            out.append(f'<text x="{left + bw + 6:.1f}" y="{y + bar_h - 4}" class="val">{fmt.format(v)}</text>')
    if note:
        out.append(f'<text x="0" y="{h - 8}" class="note">{escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def matrix_chart(title: str, arms: list[str], values: dict[str, dict[str, float]], counts: dict[str, int], ramp_var: str) -> str:
    """Arms by question, each cell a sequential-blue fill with its value printed in text ink."""
    cell_w, cell_h, left, top = 118, 30, 240, 44
    w = left + cell_w * len(QUESTIONS) + 10
    h = top + cell_h * len(arms) + 12
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}" class="chart">']
    out.append(f'<text x="0" y="18" class="title">{escape(title)}</text>')
    for j, q in enumerate(QUESTIONS):
        out.append(f'<text x="{left + cell_w * j + cell_w / 2:.1f}" y="{top - 8}" class="tick" text-anchor="middle">{q} (n{counts.get(q, 0)})</text>')
    for i, arm in enumerate(arms):
        y = top + i * cell_h
        out.append(f'<text x="{left - 10}" y="{y + cell_h - 10}" class="lab" text-anchor="end">{escape(label(arm))}</text>')
        for j, q in enumerate(QUESTIONS):
            v = values.get(arm, {}).get(q)
            if v is None:
                continue
            step = min(6, int(v * 6.999))
            x = left + cell_w * j
            out.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="4" fill="var({ramp_var}{step})">'
                f"<title>{escape(label(arm))}, {q}: {v:.2f}</title></rect>"
            )
            ink = "var(--text)" if step < 3 else "var(--surface)"
            out.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h - 10}" class="val" text-anchor="middle" fill="{ink}">{v:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def ladder_chart(title: str, ladders: list[tuple[str, float, float, float, float]]) -> str:
    """One row per ladder: a bar for the share decided alone and right, with hand-off and confidently-wrong rates as labels."""
    rows = sorted(ladders, key=lambda l: -l[4])
    w, left, bar_h, gap, top = 1320, 640, 18, 12, 44
    h = top + len(rows) * (bar_h + gap) + 30
    plot_w = w - left - 330
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}" class="chart wide">']
    out.append(f'<text x="0" y="18" class="title">{escape(title)}</text>')
    out.append(f'<text x="{left}" y="{top - 10}" class="tick">decided alone and right (share of all scored decisions)</text>')
    out.append(f'<text x="{left + plot_w + 56}" y="{top - 10}" class="tick">handed to a person; confidently wrong</text>')
    for k in range(5):
        x = left + plot_w * k / 4
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{h - 24}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{k / 4:g}</text>')
    for i, (name, human, right, cw, alone) in enumerate(rows):
        y = top + i * (bar_h + gap)
        bw = alone * plot_w
        out.append(f'<text x="{left - 10}" y="{y + bar_h - 4}" class="lab" text-anchor="end">{escape(name)}</text>')
        out.append(
            f'<path d="M{left},{y} h{max(bw - 4, 0):.1f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h-{max(bw - 4, 0):.1f} z" fill="var(--s1)">'
            f"<title>{escape(name)}: decided alone and right {alone:.3f}, to a person {human:.2f}, right incl. the person {right:.3f}, confidently wrong {cw:.3f}</title></path>"
        )
        out.append(f'<text x="{left + bw + 6:.1f}" y="{y + bar_h - 4}" class="val">{alone:.2f}</text>')
        out.append(f'<text x="{left + plot_w + 56}" y="{y + bar_h - 4}" class="val">to a person {human:.2f}; conf. wrong {cw:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def build(report: dict, title: str) -> str:
    arms = report["arms"]
    order = sorted(arms, key=lambda a: -arms[a]["accuracy"])
    counts = {q: next(iter(arms.values()))["by_question"][q]["n"] for q in QUESTIONS if q in next(iter(arms.values()))["by_question"]}
    right = [(label(a), arms[a]["accuracy"]) for a in order]
    right_iv = {label(a): tuple(arms[a]["interval"]["accuracy"]) for a in order if "interval" in arms[a]}
    cw = [(label(a), arms[a]["confidently_wrong"]) for a in sorted(arms, key=lambda a: arms[a]["confidently_wrong"])]
    cw_iv = {label(a): tuple(arms[a]["interval"]["confidently_wrong"]) for a in order if "interval" in arms[a]}
    ms = [(label(a), arms[a]["latency_ms"]) for a in sorted(arms, key=lambda a: arms[a]["latency_ms"])]
    ladders = []
    for name, rep in report.get("ladders", {}).items():
        if "[per rung and question]" not in name:
            continue
        family, _, chain = name.replace("ladder: ", "").replace(" [per rung and question]", "").rpartition(": ")
        rungs = [("a person" if r == "human" else label(r)) for r in chain.split(" > ")]
        short = (family.split(" ")[0] + ": " if family else "") + ", then ".join(rungs)
        ladders.append((short, rep["human_rate"], rep["accuracy"], rep["confidently_wrong"], rep.get("alone_right", 0.0)))
    acc_matrix = {a: {q: arms[a]["by_question"][q]["accuracy"] for q in QUESTIONS if q in arms[a]["by_question"]} for a in order}
    cw_matrix = {a: {q: arms[a]["by_question"][q]["confidently_wrong"] for q in QUESTIONS if q in arms[a]["by_question"]} for a in order}
    n = report["n_test"]
    src = report.get("thresholds_from", "")
    parts = [
        f"<h1>{escape(title)}</h1>",
        f"<p class='sub'>{n} decisions scored, replayed through every arm on the same recorded frames and state text. Thresholds for the ladders were chosen on {escape(str(src))}. Confidently wrong = answered at confidence 0.9 or more and wrong. Intervals are 95 percent by-episode bootstrap.</p>",
        '<div class="row">',
        hbar_chart("Right-decision rate per arm, 95% by-episode interval", right, "--s1", intervals=right_iv),
        hbar_chart("Confidently-wrong rate per arm (lower is better), 95% interval", cw, "--s2", intervals=cw_iv),
        "</div>",
        matrix_chart("Right-decision rate by question", order, acc_matrix, counts, "--b"),
        matrix_chart("Confidently-wrong rate by question", order, cw_matrix, counts, "--b"),
        ladder_chart("Ladders: how much the robot decides alone and right, how much goes to a person", ladders),
        hbar_chart("Mean time per decision (ms, warm)", ms, "--s1", fmt="{:.0f}", vmax=(int(max(v for _, v in ms) / 100) + 1) * 100 if ms else 1.0),
        "<h2>Table view</h2>",
        table(["arm", "n", "right", "confidently wrong", "ECE", "mean confidence", "ms"], [[label(a), str(arms[a]["n"]), f"{arms[a]['accuracy']:.3f}", f"{arms[a]['confidently_wrong']:.3f}", f"{arms[a]['ece']:.3f}", f"{arms[a]['mean_confidence']:.2f}", f"{arms[a]['latency_ms']:.0f}"] for a in order]),
        table(["ladder", "decided alone and right", "to a person", "confidently wrong", "right incl. the person"], [[nme, f"{al:.3f}", f"{hmn:.2f}", f"{c:.3f}", f"{r:.3f}"] for nme, hmn, r, c, al in sorted(ladders, key=lambda l: -l[4])]),
    ]
    style = f"""
:root{{--surface:{LIGHT['surface']};--text:{LIGHT['text']};--muted:{LIGHT['muted']};--grid:{LIGHT['grid']};--s1:{LIGHT['s1']};--s2:{LIGHT['s2']};--s3:{LIGHT['s3']};--s4:{LIGHT['s4']};{''.join(f'--b{i}:{c};' for i, c in enumerate(BLUE_RAMP_LIGHT))}}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:{DARK['surface']};--text:{DARK['text']};--muted:{DARK['muted']};--grid:{DARK['grid']};--s1:{DARK['s1']};--s2:{DARK['s2']};--s3:{DARK['s3']};--s4:{DARK['s4']};{''.join(f'--b{i}:{c};' for i, c in enumerate(BLUE_RAMP_DARK))}}}}}
:root[data-theme="dark"]{{--surface:{DARK['surface']};--text:{DARK['text']};--muted:{DARK['muted']};--grid:{DARK['grid']};--s1:{DARK['s1']};--s2:{DARK['s2']};--s3:{DARK['s3']};--s4:{DARK['s4']};{''.join(f'--b{i}:{c};' for i, c in enumerate(BLUE_RAMP_DARK))}}}
body{{background:var(--surface);color:var(--text);font-family:system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1340px}}
h1{{font-size:1.4em;margin:0 0 6px}} h2{{font-size:1.1em;margin:24px 0 8px}} .sub{{color:var(--muted);max-width:900px;margin:0 0 18px}}
.row{{display:flex;flex-wrap:wrap;gap:24px}} .chart{{width:640px;max-width:100%;height:auto;display:block;margin:12px 0 4px}} .chart.wide{{width:1320px}}
.title{{font-size:14px;font-weight:600;fill:var(--text)}} .lab{{font-size:12px;fill:var(--text)}} .val{{font-size:12px;fill:var(--text)}}
.tick{{font-size:11px;fill:var(--muted)}} .axis{{font-size:12px;fill:var(--muted)}} .note{{font-size:11px;fill:var(--muted)}} .grid{{stroke:var(--grid);stroke-width:1}} .whisker{{stroke:var(--text);stroke-width:1.2}}
.legend{{display:flex;flex-wrap:wrap;gap:10px 22px;font-size:12px;color:var(--muted);margin:0 0 18px}} .key i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0 18px}} th,td{{text-align:left;padding:4px 10px;border-bottom:1px solid var(--grid)}} th{{color:var(--muted);font-weight:600}}
"""
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Handoff benchmark</title><style>{style}</style></head><body>{''.join(parts)}</body></html>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--title", default="Decision ladder benchmark")
    args = parser.parse_args()
    log = Path(args.log)
    report = json.loads((log / "benchmark.json").read_text(encoding="utf-8"))
    out = log / "charts.html"
    out.write_text(build(report, args.title), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
