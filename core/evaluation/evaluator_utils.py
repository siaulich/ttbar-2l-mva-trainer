"""Base classes and utilities for model evaluation."""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plots."""

    figsize: Tuple[int, int] = (10, 6)
    confidence: float = 0.95
    n_bootstrap: int = 10
    show_errorbar: bool = True
    alpha: float = 0.3
    xlims: Optional[Tuple[float, float]] = None
    ylims: Optional[Tuple[float, float]] = None
    legend_loc: str = "center right"


class BootstrapCalculator:
    """Handles bootstrap calculations for confidence intervals."""

    @staticmethod
    def compute_bootstrap_ci(
        data: np.ndarray,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for a statistic.

        Args:
            data: Input data array (resampling along first dimension)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            statistic_fn: Function to compute statistic (default: mean)

        Returns:
            Tuple of (mean_statistic, lower_bound, upper_bound)
            Each element has the same shape as the statistic result
        """
        n_samples = data.shape[0]

        # Compute the first statistic to determine its shape

        bootstrap_statistics = np.zeros((n_bootstrap, *data.shape[1:]))

        for i in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_sample = data[indices]
            bootstrap_statistics[i] = np.mean(bootstrap_sample, axis=0)

        mean_statistic = np.mean(bootstrap_statistics, axis=0)
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_statistics, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_statistics, upper_percentile, axis=0)

        return mean_statistic, lower_bound, upper_bound

    @staticmethod
    def compute_binned_bootstrap(
        binning_mask: np.ndarray,
        event_weights: np.ndarray,
        data: np.ndarray,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        statistic: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute binned statistics with bootstrap confidence intervals.

        Args:
            binning_mask: Boolean mask for binning (n_bins, n_events)
            event_weights: Event weights (n_events,)
            data: Data to bin (n_events,)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            statistic: Type of statistic ('mean', 'std', 'sum')

        Returns:
            Tuple of (mean_values, lower_bounds, upper_bounds) arrays
        """
        n_samples = len(data)
        n_bins = binning_mask.shape[0]
        bootstrap_binned_values = np.zeros((n_bootstrap, n_bins))

        for i in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_data = data[indices]
            bootstrap_weights = event_weights[indices]
            bootstrap_mask = binning_mask[:, indices]

            binned_values = BinningUtility.compute_weighted_binned_statistic(
                bootstrap_mask, bootstrap_data, bootstrap_weights, statistic=statistic
            )
            bootstrap_binned_values[i] = binned_values

        mean_values = np.mean(bootstrap_binned_values, axis=0)

        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bounds = np.percentile(bootstrap_binned_values, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_binned_values, upper_percentile, axis=0)

        return mean_values, lower_bounds, upper_bounds

    @staticmethod
    def compute_binned_function_bootstrap(
        binning_mask: np.ndarray,
        event_weights: np.ndarray,
        data: Tuple[np.ndarray],
        function: Callable = np.mean,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        statistic: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute binned statistics with bootstrap confidence intervals.

        Args:
            binning_mask: Boolean mask for binning (n_bins, n_events)
            event_weights: Event weights (n_events,)
            data: Data to bin (n_events,)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            statistic: Type of statistic ('mean', 'std', 'sum')

        Returns:
            Tuple of (mean_values, lower_bounds, upper_bounds) arrays
        """
        n_samples = len(data[0])
        for data_i in data:
            assert len(data_i) == n_samples, "All data arrays must have the same length"
        n_bins = binning_mask.shape[0]
        bootstrap_binned_values = np.zeros((n_bootstrap, n_bins))

        for i in range(n_bootstrap):
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_data = tuple(data_i[indices] for data_i in data)
            bootstrap_weights = event_weights[indices]
            bootstrap_mask = binning_mask[:, indices]

            binned_values = tuple(
                BinningUtility.compute_weighted_binned_statistic(
                    bootstrap_mask,
                    bootstrap_data_i,
                    bootstrap_weights,
                    statistic=statistic,
                )
                for bootstrap_data_i in bootstrap_data
            )
            bootstrap_binned_values[i] = function(*binned_values)
        mean_values = np.mean(bootstrap_binned_values, axis=0)

        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bounds = np.percentile(bootstrap_binned_values, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_binned_values, upper_percentile, axis=0)

        return mean_values, lower_bounds, upper_bounds


class BinningUtility:
    """Utilities for binning data."""

    @staticmethod
    def create_bins(
        data: np.ndarray,
        n_bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Create bin edges."""
        if xlims is not None:
            return np.linspace(xlims[0], xlims[1], n_bins + 1)
        return np.linspace(np.min(data), np.max(data), n_bins + 1)

    @staticmethod
    def create_binning_mask(
        data: np.ndarray,
        bins: np.ndarray,
    ) -> np.ndarray:
        """
        Create boolean mask for binning.

        Args:
            data: Data to bin (n_events,)
            bins: Bin edges

        Returns:
            Boolean mask (n_bins, n_events)
        """
        return (data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )

    @staticmethod
    def compute_bin_centers(bins: np.ndarray) -> np.ndarray:
        """Compute bin centers from bin edges."""
        return 0.5 * (bins[:-1] + bins[1:])

    @staticmethod
    def compute_weighted_binned_statistic(
        binning_mask: np.ndarray,
        data: np.ndarray,
        weights: np.ndarray,
        statistic: str = "mean",
    ) -> np.ndarray:
        """
        Compute weighted statistic in bins.

        Args:
            binning_mask: Boolean mask for binning (n_bins, n_events)
            data: Data to bin (n_events,)
            weights: Event weights (n_events,)
            statistic: Type of statistic ('mean', 'std', 'sum')

        Returns:
            Array of statistics per bin
        """
        # Filter out NaN and inf values
        valid_mask = np.isfinite(data)
        data_clean = np.where(valid_mask, data, 0)

        # Apply valid mask to binning
        binning_mask_valid = binning_mask & valid_mask.reshape(1, -1)

        weighted_data = (
            data_clean.reshape(1, -1) * weights.reshape(1, -1) * binning_mask_valid
        )
        bin_weights = np.sum(weights.reshape(1, -1) * binning_mask_valid, axis=1)

        if statistic == "mean":
            bin_values = np.sum(weighted_data, axis=1)
            result = np.divide(
                bin_values,
                bin_weights,
                out=np.zeros_like(bin_values, dtype=float),
                where=bin_weights != 0,
            )
        elif statistic == "sum":
            result = np.sum(weighted_data, axis=1)
        elif statistic == "std":
            mean_values = BinningUtility.compute_weighted_binned_statistic(
                binning_mask, data, weights, statistic="mean"
            )
            squared_diff = (data_clean.reshape(1, -1) - mean_values.reshape(-1, 1)) ** 2
            weighted_squared_diff = (
                squared_diff * weights.reshape(1, -1) * binning_mask_valid
            )
            variance = np.divide(
                np.sum(weighted_squared_diff, axis=1),
                bin_weights,
                out=np.zeros_like(bin_weights, dtype=float),
                where=bin_weights != 0,
            )
            result = np.sqrt(variance)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        return result


class FeatureExtractor:
    """Utilities for extracting features from test data."""

    @staticmethod
    def extract_feature(
        X_test: dict,
        feature_indices: dict,
        feature_data_type: str,
        feature_name: str,
    ) -> np.ndarray:
        """
        Extract feature data from test set.

        Args:
            X_test: Test data dictionary
            feature_indices: Feature index mapping
            feature_data_type: Type of feature ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature

        Returns:
            Feature data array

        Raises:
            ValueError: If feature type or name not found
        """
        if feature_data_type not in X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in feature_indices[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )

        feature_idx = feature_indices[feature_data_type][feature_name]
        data = X_test[feature_data_type]

        if data.ndim == 2:
            return data[:, feature_idx]
        elif data.ndim == 3:
            return data[:, feature_idx, 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported "
                f"number of dimensions: {data.ndim}"
            )

    @staticmethod
    def get_event_weights(X_test: dict) -> np.ndarray:
        """Get event weights from test data."""
        n_events = X_test[list(X_test.keys())[0]].shape[0]
        return X_test.get("event_weight", np.ones(n_events))

    @staticmethod
    def get_event_indices(X_test: dict) -> np.ndarray:
        """Get event indices from test data."""
        n_events = X_test[list(X_test.keys())[0]].shape[0]
        return X_test.get("event_number", np.arange(n_events))

    @staticmethod
    def align_to_event_indices(
        X_test: dict,
        current_event_indices: np.ndarray,
        loaded_event_indices: np.ndarray,
    ) -> dict:
        """Align X_test to loaded event indices."""


class AccuracyCalculator:
    """Utilities for computing accuracy metrics."""

    @staticmethod
    def compute_accuracy(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        per_event: bool = True,
    ) -> np.ndarray:
        """
        Compute accuracy from predictions.

        Args:
            true_labels: True assignment labels
            predictions: Model predictions
            per_event: If True, return per-event accuracy; else overall

        Returns:
            Accuracy value(s)
        """
        predicted_indices = np.argmax(predictions, axis=-2)
        true_indices = np.argmax(true_labels, axis=-2)
        correct_predictions = np.all(predicted_indices == true_indices, axis=-1)

        if per_event:
            return correct_predictions.astype(float)
        return np.mean(correct_predictions)

    @staticmethod
    def compute_combinatoric_baseline(
        X_test: dict,
        padding_value: float,
        n_leptons: int = 2,
    ) -> np.ndarray:
        """Compute random assignment baseline accuracy."""
        num_jets = np.all(X_test["jet_inputs"] != padding_value, axis=-1).sum(axis=-1)
        return 1 / (num_jets * (num_jets - 1))


class SelectionAccuracyCalculator:
    """Utilities for computing selection accuracy metrics."""

    @staticmethod
    def compute_combinatoric_baseline(
        X_test: dict,
        padding_value: float,
        n_leptons: int = 2,
    ) -> np.ndarray:
        """Compute random assignment baseline accuracy."""
        num_jets = np.all(X_test["jet_inputs"] != padding_value, axis=-1).sum(axis=-1)
        return 2 / (num_jets * (num_jets - 1))

    @staticmethod
    def compute_selection_accuracy(
        true_labels: np.ndarray,
        predictions: np.ndarray,
        per_event: bool = True,
    ) -> np.ndarray:
        """
        Compute the accuracy of the selected jets.

        Args:
            true_labels: True assignment labels
            predictions: Model predictions
            per_event: If True, return per-event accuracy; else overall

        Returns:
            Accuracy value(s)
        """
        true_indices = np.argmax(true_labels, axis=-2)
        predicted_indices = np.argmax(predictions, axis=-2)
        correct_selections = np.all(
            predicted_indices == true_indices, axis=-1
        ) | np.all(predicted_indices[:, ::-1] == true_indices, axis=-1)
        if per_event:
            return correct_selections.astype(float)
        return np.mean(correct_selections)


class NeutrinoDeviationCalculator:
    """Utilities for computing neutrino reconstruction deviation metrics."""

    def compute_combinatoric_baseline(
        X_test: dict,
        padding_value: float,
        n_leptons: int = 2,
    ) -> np.ndarray:
        """Compute random assignment baseline accuracy."""
        num_jets = np.all(X_test["jet_inputs"] != padding_value, axis=-1).sum(axis=-1)
        return 2 / (num_jets * (num_jets - 1))

    @staticmethod
    def compute_relative_deviation(
        predicted_neutrinos: np.ndarray,
        true_neutrinos: np.ndarray,
        per_event: bool = True,
    ) -> np.ndarray:
        """
        Compute relative deviation of neutrino predictions.

        Args:
            predicted_neutrinos: Predicted neutrino momenta (n_events, 2, 3)
            true_neutrinos: True neutrino momenta (n_events, 2, 3)
            per_event: If True, return per-event deviation; else overall mean

        Returns:
            Deviation value(s)
        """
        # Compute L2 norm of the difference for each neutrino
        # Shape: (n_events, 2)
        diff_norm = np.linalg.norm(predicted_neutrinos - true_neutrinos, axis=-1)

        # Compute L2 norm of true neutrinos for normalization
        # Shape: (n_events, 2)
        true_norm = np.linalg.norm(true_neutrinos, axis=-1)

        # Compute relative deviation per neutrino
        # Shape: (n_events, 2)
        relative_dev = np.divide(
            diff_norm,
            true_norm,
            out=np.zeros_like(diff_norm, dtype=float),
            where=true_norm != 0,
        )

        # Average over both neutrinos
        # Shape: (n_events,)
        per_event_deviation = np.mean(relative_dev, axis=-1)

        if per_event:
            return per_event_deviation
        return np.mean(per_event_deviation)

    @staticmethod
    def compute_absolute_deviation(
        predicted_neutrinos: np.ndarray,
        true_neutrinos: np.ndarray,
        per_event: bool = True,
    ) -> np.ndarray:
        """
        Compute absolute deviation of neutrino predictions.

        Args:
            predicted_neutrinos: Predicted neutrino momenta (n_events, 2, 3)
            true_neutrinos: True neutrino momenta (n_events, 2, 3)
            per_event: If True, return per-event deviation; else overall mean

        Returns:
            Deviation value(s)
        """
        diff_norm = np.linalg.norm(predicted_neutrinos - true_neutrinos, axis=-1)

        per_event_deviation = np.mean(diff_norm, axis=-1)

        if per_event:
            return per_event_deviation
        return np.mean(per_event_deviation)
