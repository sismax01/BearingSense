# =============================================================================
# BearingSense — Explainable Industrial Bearing Fault Diagnostics
# Author / Signature : SISMAX (sismax01)
# Domain             : Automatic Control, Digital Signal Processing & XAI
# =============================================================================
"""BearingSense: Explainable Machine Learning for Industrial Bearing Fault Diagnosis."""

__version__ = "0.1.0"
__author__ = "SISMAX (sismax01)"

from bearingsense.bearing_kinematics import (
    BearingSpec,
    SKF_6205_DRIVE_END,
    SKF_6203_FAN_END,
    compute_fault_frequencies,
)
from bearingsense.signal_processing import (
    bandpass_filter,
    compute_envelope_spectrum,
    compute_power_spectrum,
)
from bearingsense.features import (
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    extract_features_from_window,
    extract_dataset_features,
)

__all__ = [
    "__version__",
    "__author__",
    "BearingSpec",
    "SKF_6205_DRIVE_END",
    "SKF_6203_FAN_END",
    "compute_fault_frequencies",
    "bandpass_filter",
    "compute_envelope_spectrum",
    "compute_power_spectrum",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "extract_features_from_window",
    "extract_dataset_features",
]
