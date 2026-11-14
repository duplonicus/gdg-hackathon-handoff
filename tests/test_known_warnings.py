"""Tests for the known-warning filter."""

from __future__ import annotations

import logging
import warnings

from mh3_genesis.known_warnings import KNOWN_ASSET_WARNINGS, KnownAssetWarningFilter, silence_library_deprecations


def _record(message: str, level: int = logging.WARNING) -> logging.LogRecord:
    return logging.LogRecord("genesis", level, __file__, 1, message, None, None)


def test_known_warnings_are_dropped_and_counted() -> None:
    flt = KnownAssetWarningFilter()
    for known in KNOWN_ASSET_WARNINGS:
        assert flt.filter(_record("[Genesis] " + known + " (details)")) is False
    assert sum(flt.dropped.values()) == len(KNOWN_ASSET_WARNINGS)


def test_other_warnings_and_errors_pass() -> None:
    flt = KnownAssetWarningFilter()
    assert flt.filter(_record("Something new went wrong")) is True
    assert flt.filter(_record(KNOWN_ASSET_WARNINGS[0], level=logging.ERROR)) is True
    assert flt.dropped == {}


def test_library_deprecations_are_silenced() -> None:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        silence_library_deprecations()
        warnings.warn("`torch.jit.script` is deprecated. Please switch to `torch.compile`.", FutureWarning, stacklevel=1)
        warnings.warn("keyword.__init__ got an unexpected keyword argument 'ctx'. Support is deprecated", DeprecationWarning, stacklevel=1)
        warnings.warn("some other deprecation", DeprecationWarning, stacklevel=1)
    assert [str(w.message) for w in seen] == ["some other deprecation"]
