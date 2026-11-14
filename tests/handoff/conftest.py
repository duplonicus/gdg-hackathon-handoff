"""Make the handoff package importable without installing it (the kit only installs itself)."""

import sys
from pathlib import Path

src = Path(__file__).resolve().parents[2] / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
