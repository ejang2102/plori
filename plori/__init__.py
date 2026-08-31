"""PLoRI — Per-pixel Local Rest-referenced Intensity.

A label-free, drift-robust method that extracts a cardiac contraction waveform
directly from a brightfield organoid video. See `core.py`
for the method and `README.md` for background.
"""
from plori.core import (
    segment,
    plori_signal,
    plori_perpixel,
    aggregate_for,
    derive,
    beat_metrics,
    analyze,
    autocorr_period,
)
from plori.flow import masked_flow_speed, beat_mechanics

__all__ = [
    "segment",
    "plori_signal",
    "plori_perpixel",
    "aggregate_for",
    "derive",
    "beat_metrics",
    "analyze",
    "autocorr_period",
    "masked_flow_speed",
    "beat_mechanics",
]

__version__ = "1.0.3"
