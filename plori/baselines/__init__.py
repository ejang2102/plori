"""Baseline contraction-signal methods, re-implemented for head-to-head comparison.

- `musclemotion`  — fixed single-reference frame difference (MuscleMotion).
- `contractionwave` — dense (Farneback) optical-flow magnitude (ContractionWave).

Both are faithful ports of the published methods, kept here so the comparisons in
the PLoRI paper can be reproduced from the same input videos.
"""
