"""Known, harmless warnings from the libraries and the MH3 asset, and the switches that keep them quiet.

Everything here was traced to its source on Oct 11, 2026 (Genesis 1.4.3, torch 2.14.1):

* ``torch.jit.script`` FutureWarning at ``import genesis``: Genesis's own ``utils/geom.py`` and sensor
  modules still decorate with ``torch.jit.script``; upstream main has already dropped it, no release yet.
  Nothing in this project uses TorchScript.
* ``omniverse-kit not found. USD baking will be disabled``: baking only matters for USD materials Genesis
  cannot parse itself (MDL), and every MH3 material parses to a colour or image texture. omniverse-kit is a
  multi-gigabyte NVIDIA download that would change nothing here.
* ``Mesh is not watertight. Falling back to convex hull for estimating inertial properties``: four of the
  MH3 collision meshes inside ``MH3.usd`` are open surfaces; Genesis estimates their inertia from the hull.
* ``Link '/MH3/head_camera_base_frame' has dubious center of mass``: the centre of mass authored in
  ``MH3.usd`` for that link sits outside its geometry's bounding box. An asset property, harmless for a
  camera mount.
* ``Filtered out geometry pairs causing self-collision for the neutral configuration``: MH3's collision
  hulls overlap at the home pose and Genesis disables those pairs itself, which is what we want.

The asset is licensed for this hackathon only and is not ours to edit, so these are filtered rather than fixed.
Anything not listed here still comes through.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

KNOWN_ASSET_WARNINGS: tuple[str, ...] = (
    "omniverse-kit not found. USD baking will be disabled",
    "Mesh is not watertight. Falling back to convex hull for estimating inertial properties",
    "Link '/MH3/head_camera_base_frame' has dubious center of mass",
    "Filtered out geometry pairs causing self-collision for the neutral configuration",
)


class KnownAssetWarningFilter(logging.Filter):
    """Drops Genesis WARNING records whose message starts with one of the known texts; counts what it dropped."""

    def __init__(self, known: tuple[str, ...] = KNOWN_ASSET_WARNINGS) -> None:
        super().__init__("known-asset-warnings")
        self.known = known
        self.dropped: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        message = record.getMessage()
        for known in self.known:
            if known in message:
                self.dropped[known] = self.dropped.get(known, 0) + 1
                return False
        return True


def silence_library_deprecations() -> None:
    """Hide the deprecation warnings raised inside Genesis and its dependencies at import time.

    Called by ``mh3_genesis/__init__.py`` unless MH3_SHOW_LIBRARY_WARNINGS is set, so it runs before any
    ``import genesis``. Only the two messages below are hidden; our own code does not use either API.
    """
    warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    # quadrants (Genesis's Taichi fork) builds ``ast.keyword(ctx=...)``, which Python 3.13 deprecates.
    warnings.filterwarnings("ignore", message=r"keyword\.__init__ got an unexpected keyword argument 'ctx'", category=DeprecationWarning)


def install_known_warning_filter(gs: Any) -> KnownAssetWarningFilter:
    """Attach the known-warning filter to the Genesis logger (after ``gs.init``) and return it."""
    known_filter = KnownAssetWarningFilter()
    gs.logger.addFilter(known_filter)
    return known_filter
