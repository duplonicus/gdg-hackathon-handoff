"""Run a notebook's code cells in order and save their outputs into it (a stand-in for `jupyter nbconvert --execute`).

The repo's venv has no Jupyter, nbformat or nbconvert, and adding them is not needed for one public notebook: this
reads the .ipynb as JSON, runs each code cell in one shared namespace with the notebook's folder as the working
directory (as Jupyter does), and writes back what the cell printed, the value of its last expression, and every
matplotlib figure it left open (as PNG, as Jupyter's inline backend shows them). A cell that raises stops the run and
nothing is written. With --figdir, each figure is also saved there as <figure label>.png (label from
``plt.figure(num="...")`` or ``plt.subplots(num="...")``).

    .venv\\Scripts\\python.exe scripts\\execute_notebook.py notebooks\\decision_ladder_results.ipynb --figdir media\\gallery
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lines(text: str) -> list[str]:
    """Jupyter stores multi-line strings as a list of lines, each keeping its newline."""
    return text.splitlines(keepends=True)


def run_cell(src: str, ns: dict, label: str) -> tuple[str, object]:
    tree = ast.parse(src, filename=label)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    buf = io.StringIO()
    value = None
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, label, "exec"), ns)  # noqa: S102 - running the notebook is the point
        if last is not None:
            value = eval(compile(ast.Expression(last.value), label, "eval"), ns)
    return buf.getvalue(), value


def figures(figdir: Path | None) -> list[dict]:
    out = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        if figdir is not None and fig.get_label():
            fig.savefig(figdir / f"{fig.get_label()}.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        out.append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"), "text/plain": [repr(fig)]},
            }
        )
        plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook")
    p.add_argument("--figdir", default=None, help="also save each labelled figure here as PNG")
    a = p.parse_args()
    path = Path(a.notebook).resolve()
    figdir = Path(a.figdir).resolve() if a.figdir else None
    if figdir is not None:
        figdir.mkdir(parents=True, exist_ok=True)
    nb = json.loads(path.read_text(encoding="utf-8"))
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    ns: dict = {"__name__": "__main__"}
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        text, value = run_cell("".join(cell["source"]), ns, f"<cell {count}>")
        outputs: list[dict] = []
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": lines(text)})
        if value is not None:
            outputs.append(
                {
                    "output_type": "execute_result",
                    "execution_count": count,
                    "metadata": {},
                    "data": {"text/plain": lines(repr(value))},
                }
            )
        outputs.extend(figures(figdir))
        cell["outputs"] = outputs
        cell["execution_count"] = count
        print(f"cell {count}: ok ({len(outputs)} outputs)", file=sys.stderr)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ran {count} code cells, outputs saved in {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
