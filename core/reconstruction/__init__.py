"""
Refactored evaluation module for ML-based reconstruction.

This module provides clean, modular tools for evaluating event reconstruction
methods with support for:
- Feature importance analysis
- Accuracy metrics with bootstrap confidence intervals
- Binned performance analysis
- Complementarity analysis between methods
- Mass resolution calculations
- Comprehensive visualization tools
"""

from .baseline_methods import DeltaRAssigner, ChiSquareAssigner
from .ground_truth_reconstructor import (
    GroundTruthReconstructor,
    PerfectAssignmentReconstructor,
)
from .keras_ff_reco_base import KerasFFRecoBase
from .keras_binned_regressor import KerasBinnedRegressor


__all__ = [
    # Main evaluators
    "MLEvaluator",
    "ReconstructionPlotter",
    # Feature importance
    "FeatureImportanceCalculator",
    # Base utilities
    "PlotConfig",
    "BootstrapCalculator",
    "BinningUtility",
    "FeatureExtractor",
    "AccuracyCalculator",
    # Plotting utilities
    "AccuracyPlotter",
    "ConfusionMatrixPlotter",
    "ResolutionPlotter",
    # Physics calculations
    "TopReconstructor",
    "ResolutionCalculator",
]


def get_reconstructor(reconstructor_type):

    if reconstructor_type not in globals():
        raise ValueError(f"Unknown reconstructor type: {reconstructor_type}")
    reconstructor_class = globals()[reconstructor_type]
    return reconstructor_class
