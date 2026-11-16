"""Lesion-flag network for paired (left/right) breast bioimpedance spectra.

Separate from the ``breastsim`` simulator (which stays ML-free): consumes the dataset written by
``scripts/generate_dataset.py`` and trains a binary lesion-present / lesion-absent detector on left/right pairs.
See docs/nn_design.md.
"""

__version__ = "0.1.0"
