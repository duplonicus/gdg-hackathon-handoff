"""Genesis-based digital twin for the Mirsee Robotics MH3."""

import os

from .known_warnings import silence_library_deprecations

# Genesis and its dependencies raise two deprecation warnings at import time that nothing here can act on
# (listed in known_warnings.py). Hide them before any module imports genesis; MH3_SHOW_LIBRARY_WARNINGS=1 shows them.
if not os.environ.get("MH3_SHOW_LIBRARY_WARNINGS"):
    silence_library_deprecations()

from .cli import main
from .config import CameraConfig, SimulatorConfig
from .control_panel import MH3ControlPanel
from .simulator import CameraFrame, JointState, MH3Simulation

__all__ = [
    "CameraConfig",
    "CameraFrame",
    "JointState",
    "MH3ControlPanel",
    "MH3Simulation",
    "SimulatorConfig",
    "main",
]
