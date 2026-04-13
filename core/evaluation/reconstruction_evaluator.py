"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from typing import Union, Optional, List, Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib as mpl
import atlas_mpl_style as ampl

ampl.use_atlas_style()
import matplotlib as mpl

mpl.rcParams["figure.constrained_layout.use"] = True

import os
import timeit
import keras as keras
from core.reconstruction import (
    GroundTruthReconstructor,
    KerasFFRecoBase,
)
import seaborn as sns
from ..base_classes import KerasMLWrapper, EventReconstructorBase
from .evaluator_utils import (
    PlotConfig,
    BootstrapCalculator,
    BinningUtility,
    Binning2DUtility,
    FeatureExtractor,
    AccuracyCalculator,
    SelectionAccuracyCalculator,
    NeutrinoDeviationCalculator,
)
from .plotting_utils import (
    convert_reco_name,
    BarPlotter,
    ConfusionMatrixPlotter,
    ResolutionPlotter,
    DistributionPlotter,
    BinnedFeaturePlotter,
)
from ..configs import DataConfig, BinningVariableConfig

from .physics_calculations import (
    ResolutionCalculator,
    c_hel,
    c_han,
)
from .reco_variable_config import reconstruction_variable_configs

from ..utils import (
    compute_pt_from_lorentz_vector_array,
    project_vectors_onto_axis,
    lorentz_vector_from_PtEtaPhiE_array,
    lorentz_vector_from_neutrino_momenta_array,
    scale_axis_tick_labels,
    center_axis_ticks,
)


class PredictionManager:
    """Manages predictions from multiple reconstructors."""

    def __init__(
        self,
        reconstructors: Union[EventReconstructorBase, List[EventReconstructorBase]],
        X_test: dict,
        y_test: dict,
        load_directory: Optional[str] = None,
    ):
        # Handle single reconstructor
        if isinstance(reconstructors, EventReconstructorBase):
            reconstructors = [reconstructors]

        self.reconstructors = reconstructors
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = []
        if load_directory is not None:
            self.load_predictions(load_directory)
        else:
            self._compute_all_predictions()

    def _compute_all_predictions(self):
        """Compute predictions for all reconstructors."""
        for reconstructor in self.reconstructors:
            if isinstance(reconstructor, KerasFFRecoBase):
                assignment_pred, neutrino_regression = (
                    reconstructor.complete_forward_pass(self.X_test)
                )
                self.predictions.append(
                    {
                        "assignment": assignment_pred,
                        "regression": neutrino_regression,
                    }
                )
            else:
                assignment_pred = reconstructor.predict_indices(self.X_test)
                if hasattr(reconstructor, "reconstruct_neutrinos"):
                    neutrino_pred = reconstructor.reconstruct_neutrinos(self.X_test)
                else:
                    print(
                        "WARNING: Reconstructor does not support neutrino regression."
                    )
                    neutrino_pred = None
                self.predictions.append(
                    {
                        "assignment": assignment_pred,
                        "regression": neutrino_pred,
                    }
                )
            keras.backend.clear_session(free_memory=True)

    def save_predictions(self, output_dir: str):
        """Save predictions to the specified output directory."""
        os.makedirs(output_dir, exist_ok=True)

        for idx, reconstructor in enumerate(self.reconstructors):
            reconstructor_dir = os.path.join(
                output_dir, reconstructor.get_full_reco_name().replace(" ", "_")
            )
            os.makedirs(reconstructor_dir, exist_ok=True)

            assignment_path = os.path.join(
                reconstructor_dir, "assignment_predictions.npz"
            )
            np.savez(
                assignment_path,
                predictions=self.predictions[idx]["assignment"],
            )

            if self.predictions[idx]["regression"] is not None:
                regression_path = os.path.join(
                    reconstructor_dir, "neutrino_regression_predictions.npz"
                )
                np.savez(
                    regression_path,
                    predictions=self.predictions[idx]["regression"],
                )

        np.savez(
            os.path.join(output_dir, "event_indices.npz"),
            event_indices=FeatureExtractor.get_event_indices(self.X_test),
        )

    def load_predictions(self, input_dir: str):
        """Load predictions from the specified input directory."""
        for idx, reconstructor in enumerate(self.reconstructors):
            reconstructor_dir = os.path.join(
                input_dir, reconstructor.get_full_reco_name().replace(" ", "_")
            )

            assignment_path = os.path.join(
                reconstructor_dir, "assignment_predictions.npz"
            )
            assignment_data = np.load(assignment_path)
            assignment_predictions = assignment_data["predictions"]

            regression_path = os.path.join(
                reconstructor_dir, "neutrino_regression_predictions.npz"
            )
            if os.path.exists(regression_path):
                regression_data = np.load(regression_path)
                regression_predictions = regression_data["predictions"]
            else:
                regression_predictions = None

            self.predictions.append(
                {
                    "assignment": assignment_predictions,
                    "regression": regression_predictions,
                }
            )
        event_indices_path = os.path.join(input_dir, "event_indices.npz")
        event_indices_data = np.load(event_indices_path)
        loaded_event_indices = event_indices_data["event_indices"]
        if "event_number" not in self.X_test:
            raise ValueError(
                "Event indices not found in X_test. " "Cannot align loaded predictions."
            )
        current_event_indices = FeatureExtractor.get_event_indices(self.X_test)
        if not np.array_equal(loaded_event_indices, current_event_indices):
            shared_event_indicies = np.union1d(
                loaded_event_indices, current_event_indices
            )

    def get_assignment_predictions(self, index: int) -> np.ndarray:
        """Get assignment predictions for a specific reconstructor."""
        return self.predictions[index]["assignment"]

    def get_neutrino_predictions(self, index: int) -> np.ndarray:
        """Get neutrino predictions for a specific reconstructor."""
        return self.predictions[index]["regression"]

    def get_assignment_truth(self) -> np.ndarray:
        """Get true assignment indices from y_test."""
        return self.y_test["assignment"]

    def get_neutrino_truth(self) -> Optional[np.ndarray]:
        """Get true neutrino regression targets from y_test, if available."""
        return self.y_test.get("regression", None)


class ReconstructionVariableHandler:
    """Handles configuration for reconstructed physics variables."""

    def __init__(
        self, variable_config, prediction_manager: PredictionManager, X_test: dict
    ):
        self.reco_variable_cache = {}
        self.prediction_manager = prediction_manager
        self.X_test = X_test
        self.configs = variable_config

    def compute_reconstructed_variable(
        self,
        reconstructor_index: int,
        variable_name: str,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute any reconstructed variable from event kinematics.

        Args:
            reconstructor_index: Index of the reconstructor
            variable_func: Function that takes (top1_p4, top2_p4, lepton_inputs, jet_inputs, neutrino_pred)
                          and returns the reconstructed variable(s)
            truth_extractor: Optional function to extract truth values from X_test

        Returns:
            Reconstructed variable array or tuple of (reconstructed, truth) if truth_extractor provided
        """
        if variable_name not in self.configs:
            raise ValueError(f"Variable '{variable_name}' not found in configurations.")

        if (
            variable_name in self.reco_variable_cache
            and reconstructor_index in self.reco_variable_cache[variable_name]
        ):
            return self.reco_variable_cache[variable_name][reconstructor_index]

        variable_func = self.configs[variable_name]["compute_func"]

        lepton_4vec, jet_4vecs, neutrino_4vec = self.get_reconstructed_4vectors(
            reconstructor_index
        )

        reconstructed = variable_func(lepton_4vec, jet_4vecs, neutrino_4vec)

        if isinstance(reconstructed, tuple):
            reconstructed = np.concatenate(reconstructed)

        if variable_name not in self.reco_variable_cache:
            self.reco_variable_cache[variable_name] = {}

        self.reco_variable_cache[variable_name][reconstructor_index] = reconstructed

        return reconstructed

    def select_jets(self, jet_inputs, assignment_pred):
        selected_jet_indices = assignment_pred.argmax(axis=-2)
        reco_jets = np.take_along_axis(
            jet_inputs,
            selected_jet_indices[:, :, np.newaxis],
            axis=1,
        )
        return reco_jets

    def get_jet_4vectors(self, jet_inputs, assignment_pred):
        true_jets = self.select_jets(jet_inputs, assignment_pred)
        return lorentz_vector_from_PtEtaPhiE_array(true_jets[..., :4])

    def get_reconstructed_4vectors(self, reconstructor_index):
        assignment_pred = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        neutrino_pred = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )
        lepton_inputs = self.X_test["lep_inputs"]
        jet_inputs = self.X_test["jet_inputs"][:, :, :4]

        lepton_4vec = lorentz_vector_from_PtEtaPhiE_array(lepton_inputs)
        neutrino_4vec = lorentz_vector_from_neutrino_momenta_array(neutrino_pred)
        jet_4vecs = self.get_jet_4vectors(jet_inputs, assignment_pred)

        return lepton_4vec, jet_4vecs, neutrino_4vec

    def compute_true_variable(
        self,
        variable_name: str,
    ) -> np.ndarray:
        """
        Compute any truth variable from X_test.

        Args:
            truth_extractor: Function to extract truth values from X_test
        Returns:
            Truth variable array
        """
        if variable_name not in self.configs:
            raise ValueError(f"Variable '{variable_name}' not found in configurations.")

        truth_cache_key = f"{variable_name}_truth"
        if truth_cache_key in self.reco_variable_cache:
            return self.reco_variable_cache[truth_cache_key]

        truth_extractor = self.configs[variable_name]["extract_func"]

        truth = truth_extractor(self.X_test)

        if isinstance(truth, tuple):
            truth = np.concatenate(truth)

        truth_cache_key = f"{variable_name}_truth"
        self.reco_variable_cache[truth_cache_key] = truth

        return truth


class ReconstructionPlotter:
    """Evaluator for comparing event reconstruction methods."""

    def __init__(
        self,
        prediction_manager: PredictionManager,
        reco_variable_configs: dict = reconstruction_variable_configs,
    ):
        self.variable_configs = reco_variable_configs
        self.prediction_manager = prediction_manager

        self.X_test = prediction_manager.X_test
        self.y_test = prediction_manager.y_test

        self.config = self.prediction_manager.reconstructors[0].config

        self.variable_handler = ReconstructionVariableHandler(
            reco_variable_configs, prediction_manager, self.X_test
        )

    def _validate_configs(self):
        """Validate that all reconstructors have the same configuration."""
        configs = [r.config for r in self.prediction_manager.reconstructors]
        base_config = configs[0]

        for config in configs[1:]:
            if config != base_config:
                raise ValueError(
                    "All reconstructors must have the same DataConfig for "
                    "consistent evaluation."
                )

    # ==================== Accuracy Methods ====================

    def evaluate_accuracy(
        self,
        reconstructor_index: int,
        per_event: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evaluate accuracy for a specific reconstructor."""
        predictions = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        return AccuracyCalculator.compute_accuracy(
            self.y_test["assignment"],
            predictions,
            per_event=per_event,
        )

    def evaluate_selection_accuracy(
        self,
        reconstructor_index: int,
        per_event: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evaluate selection accuracy for a specific reconstructor."""
        predictions = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        return SelectionAccuracyCalculator.compute_selection_accuracy(
            self.y_test["assignment"],
            predictions,
            per_event=per_event,
        )

    def _bootstrap_accuracy(
        self,
        reconstructor_index: int,
        config: PlotConfig,
    ) -> Tuple[float, float, float]:
        """Compute accuracy with bootstrap confidence intervals."""
        accuracy_data = self.evaluate_accuracy(reconstructor_index, per_event=True)
        return BootstrapCalculator.compute_bootstrap_ci(
            accuracy_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def _bootstrap_selection_accuracy(
        self,
        reconstructor_index: int,
        config: PlotConfig,
    ) -> Tuple[float, float, float]:
        """Compute selection accuracy with bootstrap confidence intervals."""
        accuracy_data = self.evaluate_selection_accuracy(
            reconstructor_index, per_event=True
        )
        return BootstrapCalculator.compute_bootstrap_ci(
            accuracy_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def plot_all_accuracies(
        self,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot accuracies for all reconstructors with error bars."""
        config = PlotConfig(
            figsize=figsize,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        accuracies = []

        names = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            mean_acc, lower, upper = self._bootstrap_accuracy(i, config)
            accuracies.append((mean_acc, lower, upper))
            names.append(reconstructor.get_assignment_name())

        return BarPlotter.plot_bar_chart(
            names, accuracies, "Assignment Accuracy", config
        )

    def plot_all_selection_accuracies(
        self,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot accuracies for all reconstructors with error bars."""
        config = PlotConfig(
            figsize=figsize,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        selection_accuracies = []

        names = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            mean_acc, lower, upper = self._bootstrap_selection_accuracy(i, config)
            selection_accuracies.append((mean_acc, lower, upper))
            names.append(reconstructor.get_assignment_name())

        return BarPlotter.plot_bar_chart(
            names, selection_accuracies, "Selection Accuracy", config
        )

    # ==================== Neutrino Deviation Methods ====================

    def evaluate_neutrino_deviation(
        self,
        reconstructor_index: int,
        per_event: bool = False,
        deviation_type: str = "relative",
    ) -> Union[float, np.ndarray]:
        """
        Evaluate neutrino reconstruction deviation for a specific reconstructor.

        Args:
            reconstructor_index: Index of the reconstructor
            per_event: If True, return per-event deviation; else overall mean
            deviation_type: Type of deviation ('relative' or 'absolute')

        Returns:
            Deviation value(s)
        """
        if self.y_test.get("regression") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        predictions = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )
        true_neutrinos = self.y_test["regression"]

        if deviation_type == "relative":
            return NeutrinoDeviationCalculator.compute_relative_deviation(
                predictions,
                true_neutrinos,
                per_event=per_event,
            )
        elif deviation_type == "absolute":
            return NeutrinoDeviationCalculator.compute_absolute_deviation(
                predictions,
                true_neutrinos,
                per_event=per_event,
            )
        else:
            raise ValueError(
                f"Unknown deviation_type: {deviation_type}. "
                "Must be 'relative' or 'absolute'."
            )

    def _bootstrap_neutrino_deviation(
        self,
        reconstructor_index: int,
        config: PlotConfig,
        deviation_type: str = "relative",
    ) -> Tuple[float, float, float]:
        """Compute neutrino deviation with bootstrap confidence intervals."""
        deviation_data = self.evaluate_neutrino_deviation(
            reconstructor_index, per_event=True, deviation_type=deviation_type
        )
        return BootstrapCalculator.compute_bootstrap_ci(
            deviation_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def plot_all_neutrino_deviations(
        self,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
        deviation_type: str = "relative",
    ):
        """
        Plot neutrino deviations for all reconstructors with error bars.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            figsize: Figure size
            deviation_type: Type of deviation ('relative' or 'absolute')

        Returns:
            Tuple of (figure, axis)
        """
        if self.y_test.get("regression") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        config = PlotConfig(
            figsize=figsize,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        deviations = []
        names = []

        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):

            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                continue

            mean_dev, lower, upper = self._bootstrap_neutrino_deviation(
                i, config, deviation_type
            )
            deviations.append((mean_dev, lower, upper))
            names.append(reconstructor.get_full_reco_name())

        if not deviations:
            raise ValueError(
                "No reconstructors with neutrino reconstruction found. "
                "Cannot plot neutrino deviations."
            )

        return BarPlotter.plot_bar_chart(
            names,
            deviations,
            f"Neutrino {deviation_type.capitalize()} Deviation",
            config,
        )

    # ==================== Binned Accuracy Methods ====================

    def plot_binned_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        show_combinatoric: bool = True,
    ):
        """Plot binned accuracy vs. a feature with bootstrap error bars."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
            ylims=(0, 1.1),
            legend_loc="upper right",
        )

        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )
        if np.isnan(feature_data).any() or np.isinf(feature_data).any():
            print(
                f"WARNING: NaN values found in feature data for {feature_name}. "
                "These will be ignored in the binned accuracy calculation."
            )

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        if np.any(~np.any(binning_mask, axis=1)):
            print(
                "WARNING: Some bins have no events. "
                "These bins will be ignored in the binned accuracy calculation."
            )

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute binned accuracies for each reconstructor
        binned_accuracies = []
        names = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            accuracy_data = self.evaluate_accuracy(i, per_event=True)

            if show_errorbar:
                mean_acc, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    accuracy_data,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_accuracies.append((mean_acc, lower, upper))
            else:
                binned_acc = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask, accuracy_data, event_weights
                )
                binned_accuracies.append((binned_acc, binned_acc, binned_acc))
            names.append(reconstructor.get_assignment_name())

        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        feature_label = fancy_feature_label or feature_name

        fig, ax = BinnedFeaturePlotter.plot_binned_feature(
            bin_centers,
            binned_accuracies,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            "Assignment Accuracy",
            config,
        )
        return fig, ax

    def plot_2d_binned_accuracy(
        self,
        feature1_binning_config: BinningVariableConfig,
        feature2_binning_config: BinningVariableConfig,
    ):
        """Plot 2D binned accuracy vs. two features."""
        config = PlotConfig(
            confidence=0.95,
            n_bootstrap=10,
            show_errorbar=False,
            legend_loc="upper right",
        )
        feature1_data_type = feature1_binning_config.feature_type
        feature1_name = feature1_binning_config.feature_name
        fancy_feature1_label = feature1_binning_config.fancy_feature_label
        bins_feature1 = feature1_binning_config.bins
        xlims_feature1 = feature1_binning_config.xlims

        feature2_data_type = feature2_binning_config.feature_type
        feature2_name = feature2_binning_config.feature_name
        fancy_feature2_label = feature2_binning_config.fancy_feature_label
        bins_feature2 = feature2_binning_config.bins
        xlims_feature2 = feature2_binning_config.xlims

        # Extract feature data
        feature1_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature1_data_type,
            feature1_name,
        )
        feature2_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature2_data_type,
            feature2_name,
        )
        if feature1_binning_config.rescale_factor is not None:
            feature1_data *= feature1_binning_config.rescale_factor
            if xlims_feature1 is not None:
                xlims_feature1 = (
                    xlims_feature1[0] * feature1_binning_config.rescale_factor,
                    xlims_feature1[1] * feature1_binning_config.rescale_factor,
                )
        if feature2_binning_config.rescale_factor is not None:
            feature2_data *= feature2_binning_config.rescale_factor
            if xlims_feature2 is not None:
                xlims_feature2 = (
                    xlims_feature2[0] * feature2_binning_config.rescale_factor,
                    xlims_feature2[1] * feature2_binning_config.rescale_factor,
                )

        # Create bins
        bin_edges_feature1, bin_edges_feature2 = Binning2DUtility.create_bins(
            feature1_data,
            feature2_data,
            bins_feature1,
            bins_feature2,
            xlims_feature1,
            xlims_feature2,
        )
        binning_mask = Binning2DUtility.create_binning_mask(
            feature1_data, feature2_data, bin_edges_feature1, bin_edges_feature2
        )

        # Mask out bins with too few events
        min_events_per_bin = 10
        bin_counts = (binning_mask != 0).sum(axis=2)
        low_stat_mask = bin_counts < min_events_per_bin
        binning_mask[low_stat_mask, ...] = False

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute binned accuracies for each reconstructor
        binned_accuracies = []
        names = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            accuracy_data = self.evaluate_accuracy(i, per_event=True)

            binned_acc = Binning2DUtility.compute_weighted_binned_statistic(
                binning_mask, accuracy_data, event_weights
            )
            binned_accuracies.append(binned_acc)
            names.append(reconstructor.get_assignment_name())

        feature1_label = fancy_feature1_label or feature1_name
        feature2_label = fancy_feature2_label or feature2_name

        fig, ax = BinnedFeaturePlotter.plot_2d_binned_feature(
            binned_accuracies,
            names,
            bin_edges_feature1,
            bin_edges_feature2,
            feature1_label,
            feature2_label,
            "Assignment Accuracy",
            config,
        )
        return fig, ax

    def plot_binned_selection_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        show_combinatoric: bool = True,
    ):
        """Plot binned accuracy vs. a feature with bootstrap error bars."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
            legend_loc="upper right",
        )

        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        combinatoric_accuracy = None
        if show_combinatoric:
            combinatoric_per_event = (
                SelectionAccuracyCalculator.compute_combinatoric_baseline(
                    self.X_test, self.config.padding_value
                )
            )
            combinatoric_accuracy = BinningUtility.compute_weighted_binned_statistic(
                binning_mask, combinatoric_per_event, event_weights
            )

        binned_accuracies = []
        names = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            accuracy_data = self.evaluate_selection_accuracy(i, per_event=True)

            if show_errorbar:
                mean_acc, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    accuracy_data,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_accuracies.append((mean_acc, lower, upper))
            else:
                binned_acc = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask, accuracy_data, event_weights
                )
                binned_accuracies.append((binned_acc, binned_acc, binned_acc))
            names.append(reconstructor.get_assignment_name())

        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        feature_label = fancy_feature_label or feature_name

        fig, ax = BinnedFeaturePlotter.plot_binned_feature(
            bin_centers,
            binned_accuracies,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            "Selection Accuracy",
            config,
        )
        return fig, ax

    def plot_binned_accuracy_quotients(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
    ):
        """
        Plot binned quotient of assignment accuracy / selection accuracy vs. a feature.

        The quotient indicates how well the assignment performs relative to just
        selecting the correct jets (regardless of assignment to leptons).

        Args:
            feature_data_type: Type of feature data ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature to bin by
            fancy_feature_label: Optional fancy label for the feature
            bins: Number of bins
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            show_errorbar: Whether to show error bars

        Returns:
            Tuple of (figure, axis)
        """
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=True,
            legend_loc="upper right",
        )

        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute binned quotients for each reconstructor
        binned_quotients = []
        names = []

        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            # Get per-event accuracies
            assignment_acc = self.evaluate_accuracy(i, per_event=True)
            selection_acc = self.evaluate_selection_accuracy(i, per_event=True)

            mean_quotient, lower, upper = (
                BootstrapCalculator.compute_binned_function_bootstrap(
                    binning_mask,
                    event_weights,
                    (assignment_acc, selection_acc),
                    lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y != 0),
                    config.n_bootstrap,
                    config.confidence,
                    statistic="mean",
                )
            )
            binned_quotients.append((mean_quotient, lower, upper))

            names.append(reconstructor.get_assignment_name())

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        feature_label = fancy_feature_label or feature_name
        fig, ax = BinnedFeaturePlotter.plot_binned_feature(
            bin_centers,
            binned_quotients,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            "Accuracy Quotient (Assignment / Selection)",
            config,
        )
        return fig, ax

    def plot_confusion_matrices(
        self,
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5),
    ):
        """Plot confusion matrices for all reconstructors."""
        predictions_list = [
            self.prediction_manager.get_assignment_predictions(i)
            for i in range(len(self.prediction_manager.reconstructors))
            if not isinstance(
                self.prediction_manager.reconstructors[i],
                GroundTruthReconstructor,
            )
        ]
        names = [
            r.get_assignment_name()
            for r in self.prediction_manager.reconstructors
            if not isinstance(r, GroundTruthReconstructor)
        ]

        return ConfusionMatrixPlotter.plot_confusion_matrices(
            self.y_test["assignment"],
            predictions_list,
            names,
            normalize,
            figsize_per_plot,
        )

    def plot_complementarity_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        """Plot complementarity matrix between reconstructors."""
        matrix = self.compute_complementarity_matrix()
        names = [
            r.get_assignment_name()
            for r in self.prediction_manager.reconstructors
            if not isinstance(r, GroundTruthReconstructor)
        ]
        return sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            xticklabels=names,
            yticklabels=names,
            cmap="viridis",
            cbar_kws={"label": "Complementarity"},
            ax=ax,
        )

    def plot_binned_reco_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        variable_name: str,
        ylabel: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        statistic: str = "std",
        use_signed_deviation: bool = True,
        use_relative_deviation: bool = False,
    ):
        """
        Plot binned resolution or deviation of any reconstructed variable vs. a feature.

        Args:
            feature_data_type: Type of feature data ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature to bin by
            variable_func: Function that takes (top1_p4, top2_p4, lepton_inputs, jet_inputs, neutrino_pred)
                          and returns reconstructed variable(s)
            truth_extractor: Function to extract truth values from (X_test, feature_indices)
            ylabel: Y-axis label for the plot
            fancy_feature_label: Optional fancy label for the feature
            bins: Number of bins
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            show_errorbar: Whether to show error bars
            statistic: Statistic to compute ('std' for resolution, 'mean' for deviation)
            use_signed_deviation: If True, use signed deviation instead of absolute

        Returns:
            Tuple of (figure, axis)
        """
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Extract feature data and create bins
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute metric for each reconstructor
        binned_metrics = []

        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            # Compute reconstructed and truth values
            reconstructed = self.variable_handler.compute_reconstructed_variable(
                i, variable_name
            )
            truth = self.variable_handler.compute_true_variable(variable_name)

            # Compute deviation
            deviation = ResolutionCalculator.compute_deviation(
                reconstructed,
                truth,
                use_signed_deviation=use_signed_deviation,
                use_relative_deviation=use_relative_deviation,
            )

            if show_errorbar:
                mean_metric, lower, upper = (
                    BootstrapCalculator.compute_binned_bootstrap(
                        binning_mask,
                        event_weights,
                        deviation,
                        config.n_bootstrap,
                        config.confidence,
                        statistic=statistic,
                    )
                )
                binned_metrics.append((mean_metric, lower, upper))
            else:
                metric = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask,
                    deviation,
                    event_weights,
                    statistic=statistic,
                )
                binned_metrics.append((metric, metric, metric))

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        names = [r.get_full_reco_name() for r in self.prediction_manager.reconstructors]

        return ResolutionPlotter.plot_binned_resolution(
            bin_centers,
            binned_metrics,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            ylabel,
            config,
        )

    def plot_reco_vs_truth_distribution(
        self,
        ax,
        reconstructor_index: int,
        variable_name: str,
        variable_label: str,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
    ):
        """
        Plot distribution of reconstructed variable vs. truth.

        Args:
            reconstructor_index: Index of the reconstructor
            variable_func: Function that takes (top1_p4, top2_p4, lepton_inputs, jet_inputs, neutrino_pred)
                          and returns reconstructed variable(s)
            truth_extractor: Function to extract truth values from X_test
            variable_label: Label for the variable being plotted
            bins: Number of bins
            xlims: Optional x-axis limits
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        # Compute reconstructed and truth values
        reconstructed = self.variable_handler.compute_reconstructed_variable(
            reconstructor_index, variable_name
        )
        config = PlotConfig(xlims=xlims)
        truth = self.variable_handler.compute_true_variable(variable_name)

        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        return DistributionPlotter.plot_feature_distributions(
            [reconstructed, truth],
            variable_label,
            event_weights=event_weights,
            bins=bins,
            labels=["reco", "truth"],
            ax=ax,
            config=config,
        )

    def plot_deviations_distributions_all_reconstructors(
        self,
        variable_name: str,
        variable_label: str,
        figsize: Optional[Tuple[int, int]] = (10, 10),
        use_signed_deviation: bool = True,
        use_relative_deviation: bool = False,
        deviation_function: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Plot distributions of deviations for all reconstructors.

        Args:
            variable_func: Function that computes the variable from (leptons, jets, neutrinos)
            truth_extractor: Function that extracts truth values from X_test
            variable_label: Label for the variable being plotted
            bins: Number of bins
            figsize: Figure size
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(
            figsize=figsize,
        )

        # Collect all deviations and labels
        all_deviations = []
        labels = []
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Extract common parameters from kwargs

        for reco_index, reconstructor in enumerate(
            self.prediction_manager.reconstructors
        ):
            # Compute reconstructed and truth values
            reconstructed = self.variable_handler.compute_reconstructed_variable(
                reco_index, variable_name
            )
            truth = self.variable_handler.compute_true_variable(variable_name)

            # Compute deviation
            deviation = ResolutionCalculator.compute_deviation(
                reconstructed,
                truth,
                use_signed_deviation=use_signed_deviation,
                use_relative_deviation=use_relative_deviation,
                deviation_function=deviation_function,
            )

            all_deviations.append(deviation)
            labels.append(reconstructor.get_full_reco_name())

            event_weights_plot = event_weights

        # Plot all deviations together
        DistributionPlotter.plot_feature_distributions(
            all_deviations,
            ("Relative " if use_relative_deviation else "")
            + f"Deviation in {variable_label}",
            event_weights=event_weights_plot,
            labels=labels,
            ax=axes,
            **kwargs,
        )

        return fig, axes

    def plot_distributions_all_reconstructors(
        self,
        variable_name: str,
        variable_label: str,
        xlims: Optional[Tuple[float, float]] = None,
        bins: int = 50,
        figsize: Optional[Tuple[int, int]] = (6, 5),
        save_individual_plots: bool = False,
        **kwargs,
    ):
        """
        Plot distributions for all reconstructors and truth.

        Args:
            variable_func: Function that computes the variable from (leptons, jets, neutrinos)
            truth_extractor: Function that extracts truth values from X_test
            variable_label: Label for the variable being plotted
            xlims: Optional x-axis limits
            bins: Number of bins
            figsize: Figure size

        Returns:
            Tuple of (figure, axes)
        """
        num_plots = len(self.prediction_manager.reconstructors)  # Exclude ground truth

        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)

        if not save_individual_plots:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(figsize[0] * num_cols, figsize[1] * num_rows),
                constrained_layout=True,
            )
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
        else:
            fig, axes = [], []
            for i in range(num_plots):
                fig_i, ax_i = plt.subplots(
                    figsize=figsize,
                    constrained_layout=True,
                )
                fig.append(fig_i)
                axes.append(ax_i)

        for reco_index, reconstructor in enumerate(
            self.prediction_manager.reconstructors
        ):
            #            if isinstance(reconstructor, GroundTruthReconstructor):
            #                continue
            ax = axes[reco_index]
            self.plot_reco_vs_truth_distribution(
                ax,
                reco_index,
                variable_name,
                variable_label,
                bins=bins,
                xlims=xlims,
                **kwargs,
            )
            ax.set_title(reconstructor.get_full_reco_name())

        if not save_individual_plots:
            for i in range(len(self.prediction_manager.reconstructors), len(axes)):
                fig.delaxes(axes[i])  # Remove unused subplots

        return fig, axes

    # ==================== Plot Specific Variable distributions ====================

    def plot_variable_distribution(self, variable_key: str, **kwargs):
        """Generic method to plot variable distributions using configuration."""
        config = self.variable_configs[variable_key]

        return self.plot_distributions_all_reconstructors(
            variable_key,
            variable_label=config["label"],
            **kwargs,
        )

    # ==================== Deviation Distribution Methods ====================

    def plot_variable_deviation(self, variable_key: str, **kwargs):
        """Generic method to plot variable deviations using configuration."""
        config = self.variable_configs[variable_key]
        # Set defaults from config, allow kwargs to override
        defaults = {
            "use_relative_deviation": (
                kwargs["use_relative_deviation"]
                if "use_relative_deviation" in kwargs
                else config.get("use_relative_deviation", False)
            ),
            "deviation_function": config.get("deviation_function", None),
        }
        if "deviation_label" in config:
            variable_label = config["deviation_label"]
        else:
            label = config["label"]
            variable_label = (
                f"Relative Deviation in {label}"
                if defaults["use_relative_deviation"]
                else f"Deviation in {label}"
            )

        defaults.update(kwargs)

        return self.plot_deviations_distributions_all_reconstructors(
            variable_name=variable_key,
            variable_label=variable_label,
            **defaults,
        )

    def plot_variable_confusion_matrix(
        self,
        variable_key: str,
        **kwargs,
    ):
        """Generic method to plot variable confusion matrices using configuration."""
        if variable_key not in self.variable_configs:
            raise ValueError(
                f"Variable key '{variable_key}' not found in configurations."
            )

        return self.plot_variable_confusion_matrix_for_all_reconstructors(
            variable_name=variable_key,
            variable_label=f"{self.variable_configs[variable_key]['label']}",
            **kwargs,
        )

    def plot_variable_confusion_matrix_for_all_reconstructors(
        self,
        variable_name: str,
        variable_label: str,
        bins: int = 10,
        xlims: Optional[Tuple[float, float]] = None,
        figsize_per_plot: Tuple[int, int] = (7, 7),
        normalize: str = "all",
        **kwargs,
    ):
        """Plot confusion matrices for all reconstructors for a specific variable."""
        names = []
        truth = self.variable_handler.compute_true_variable(variable_name)
        reconstructed_list = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if (
                isinstance(reconstructor, GroundTruthReconstructor)
                and not reconstructor.use_nu_flows
                and not reconstructor.perform_regression
            ):
                continue

            # Compute reconstructed and truth values
            reconstructed_list.append(
                self.variable_handler.compute_reconstructed_variable(i, variable_name)
            )
            names.append(reconstructor.get_full_reco_name())

        if xlims is None:
            xlims = np.min(np.concatenate([*reconstructed_list, truth])), np.max(
                np.concatenate([*reconstructed_list, truth])
            )

        # Digitize into bins
        bin_edges = np.linspace(
            xlims[0],
            xlims[1],
            bins + 1,
        )
        num_plots = len(self.prediction_manager.reconstructors)  # Exclude ground truth
        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)
        
        figures = {}

        for reco_index, reconstructed, name in zip(
            range(len(reconstructed_list)), reconstructed_list, names
        ):
            fig, ax = plt.subplots(figsize=figsize_per_plot)
            ConfusionMatrixPlotter.plot_variable_confusion_matrix(
                truth,
                reconstructed,
                variable_label,
                ax,
                bin_edges,
                normalize=normalize,
                **kwargs,
            )
            figures[name] = (fig, ax)
            # Add correlation coefficient to axis
            corr_coeff = np.corrcoef(truth, reconstructed)[0, 1]
            ampl.draw_tag(text = f"Corr: {corr_coeff:.2f}", ax=ax)
        return figures

    # ==================== Binned Variable Resolution/Deviation Methods ====================

    def plot_binned_variable(
        self,
        variable_key: str,
        metric_type: str,
        feature_data_type: str,
        feature_name: str,
        **kwargs,
    ):
        """Generic method to plot binned metrics using configuration.

        Args:
            variable_key: Key identifying the variable (e.g., 'top_mass', 'c_han')
            metric_type: Either 'resolution' or 'deviation'
            feature_data_type: Type of feature data for binning
            feature_name: Name of feature for binning
            **kwargs: Additional arguments passed to plot_binned_reco_resolution
        """
        config = self.variable_configs[variable_key]
        resolution_config = config.get("resolution", {})

        # Determine parameters based on metric type
        if metric_type == "resolution":
            statistic = "std"
            use_signed_deviation = False
            ylabel_template = resolution_config.get(
                "ylabel_resolution", f"{config['label']} Resolution"
            )
        else:  # deviation
            statistic = "mean"
            use_signed_deviation = True
            ylabel_template = resolution_config.get(
                "ylabel_deviation", f"Mean {config['label']} Deviation"
            )

        # Get defaults from config
        defaults = {
            "statistic": statistic,
            "use_signed_deviation": use_signed_deviation,
            "use_relative_deviation": resolution_config.get(
                "use_relative_deviation", True
            ),
        }
        defaults.update(kwargs)

        return self.plot_binned_reco_resolution(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_name=variable_key,
            ylabel=ylabel_template,
            **defaults,
        )

    # ==================== Variable Configuration and Computation Methods ====================

    def save_accuracy_latex_table(
        self,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Generate LaTeX table with accuracy and selection accuracy for all reconstructors.

        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence: Confidence level for intervals
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table string
        """
        config = PlotConfig(
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

        # Collect results
        results = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            name = reconstructor.get_assignment_name()

            # Compute accuracy with CI
            acc_mean, acc_lower, acc_upper = self._bootstrap_accuracy(i, config)

            # Compute selection accuracy with CI
            sel_acc_mean, sel_acc_lower, sel_acc_upper = (
                self._bootstrap_selection_accuracy(i, config)
            )

            results.append(
                {
                    "name": name,
                    "accuracy": (acc_mean, acc_lower, acc_upper),
                    "selection_accuracy": (sel_acc_mean, sel_acc_lower, sel_acc_upper),
                }
            )

        # Generate LaTeX table
        latex = []
        latex.append(r"    \begin{tabular}{lcc}")
        latex.append(r"        \toprule")
        latex.append(r"        Method & Assignment Accuracy & Selection Accuracy \\")
        latex.append(r"        \midrule")

        for res in results:
            name = res["name"]
            acc_mean, acc_lower, acc_upper = res["accuracy"]
            sel_mean, sel_lower, sel_upper = res["selection_accuracy"]

            acc_str = (
                f"${acc_mean:.4f}"
                + "_{-"
                + f"{acc_mean - acc_lower:.4f}"
                + "}"
                + "^{+"
                + f"{acc_upper - acc_mean:.4f}"
                + "}$"
            )
            sel_str = (
                f"${sel_mean:.4f}"
                + "_{-"
                + f"{sel_mean - sel_lower:.4f}"
                + "}"
                + "^{+"
                + f"{sel_upper - sel_mean:.4f}"
                + "}$"
            )

            latex.append(f"        {name} & {acc_str} & {sel_str} \\\\")

        latex.append(r"        \bottomrule")
        latex.append(r"    \end{tabular}")

        latex_str = "\n".join(latex)
        file_name = "reconstruction_accuracies_table.tex"
        if save_dir is not None:
            file_name = os.path.join(save_dir, file_name)
        with open(file_name, "w") as f:
            f.write(latex_str)

    def save_regression_error_latex_table(
        self,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Generate LaTeX table with regression errors for a specific variable across all reconstructors.

        Args:
            variable_key: Key identifying the variable (e.g., 'top_mass', 'c_han')
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence: Confidence level for intervals
            save_dir: Optional directory to save the LaTeX file
        Returns:
            LaTeX table string
        """
        components_mse = []
        names = []
        use_nu_flows = False
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if reconstructor.use_nu_flows and not use_nu_flows:
                use_nu_flows = True
            elif reconstructor.use_nu_flows and use_nu_flows:
                continue
            elif (
                isinstance(reconstructor, GroundTruthReconstructor)
                and not reconstructor.perform_regression
            ):
                continue
            names.append(reconstructor.get_neutrino_name())
            error = (
                np.abs(
                    self.prediction_manager.get_neutrino_predictions(i)
                    - self.prediction_manager.y_test["regression"]
                )
                / 1e3
            )
            mse_ci = BootstrapCalculator.compute_bootstrap_ci(
                data=error,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
            )
            components_mse.append(mse_ci)
            # Generate LaTeX table
        latex = []
        latex.append(r"    \begin{tabular}{lccc}")
        latex.append(r"        \toprule")
        latex.append(r"        Method & MSE ($p_x$) & MSE ($p_y$) & MSE ($p_z$) \\")
        latex.append(r"        \midrule")
        for name, mse_ci in zip(names, components_mse):
            mse_mean, mse_lower, mse_upper = mse_ci
            mse_top_nu_str = []
            mse_tbar_nu_str = []
            for i in range(3):
                mse_top_nu_str.append(
                    f"${mse_mean[0,i]:.1f}"
                    + "_{-"
                    + f"{mse_mean[0,i] - mse_lower[0,i]:.1f}"
                    + "}"
                    + "^{+"
                    + f"{mse_upper[0,i] - mse_mean[0,i]:.1f}"
                    + "}$"
                )
                mse_tbar_nu_str.append(
                    f"${mse_mean[1,i]:.1f}"
                    + "_{-"
                    + f"{mse_mean[1,i] - mse_lower[1,i]:.1f}"
                    + "}"
                    + "^{+"
                    + f"{mse_upper[1,i] - mse_mean[1,i]:.1f}"
                    + "}$"
                )
            latex.append(
                r"         \multirow{2}{*}{"
                + f"{name}"
                + r"}"
                + f"& {mse_top_nu_str[0]} & {mse_top_nu_str[1]} & {mse_top_nu_str[2]} \\\\"
            )
            latex.append(
                f"         & {mse_tbar_nu_str[0]} & {mse_tbar_nu_str[1]} & {mse_tbar_nu_str[2]} \\\\"
            )
            latex.append(r"        \midrule")
        latex.append(r"        \bottomrule")
        latex.append(r"    \end{tabular}")
        latex_str = "\n".join(latex)
        file_name = "neutrino_regression_errors_table.tex"
        if save_dir is not None:
            file_name = os.path.join(save_dir, file_name)
        with open(file_name, "w") as f:
            f.write(latex_str)
        return latex_str

    def save_reconstruction_variable_latex_table(
        self,
        variable_key: str,
        n_bootstrap: int = 10,
        confidence: float = 0.95,
        save_dir: Optional[str] = None,
        use_signed_deviation: bool = True,
        use_relative_deviation: bool = False,
        deviation_function: Optional[Callable] = None,
    ) -> str:
        """
        Generate LaTeX table with binned reconstruction variable metrics for all reconstructors.

        Args:
            variable_key: Key identifying the variable (e.g., 'top_mass', 'c_han')
            metric_type: Either 'resolution' or 'deviation'
            feature_data_type: Type of feature data for binning
            feature_name: Name of feature for binning
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence: Confidence level for intervals
            save_dir: Optional directory to save the LaTeX file
        Returns:
            LaTeX table string
        """

        # Collect results
        results = []
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            # Compute reconstructed and truth values
            reconstructed = self.variable_handler.compute_reconstructed_variable(
                i, variable_key
            )
            truth = self.variable_handler.compute_true_variable(variable_key)

            deviation = ResolutionCalculator.compute_deviation(
                reconstructed,
                truth,
                use_signed_deviation=use_signed_deviation,
                use_relative_deviation=use_relative_deviation,
                deviation_function=deviation_function,
            )
            square_deviation = deviation**2

            # Compute binned metric with CI
            dev_mean, dev_lower, dev_upper = BootstrapCalculator.compute_bootstrap_ci(
                data=deviation,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
            )
            square_dev_mean, square_dev_lower, square_dev_upper = (
                BootstrapCalculator.compute_bootstrap_ci(
                    data=square_deviation,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                )
            )
            results.append(
                {
                    "name": reconstructor.get_full_reco_name(),
                    "mean_dev_metric": dev_mean,
                    "dev_lower": dev_lower,
                    "dev_upper": dev_upper,
                    "mean_square_metric": square_dev_mean,
                    "lower_square": square_dev_lower,
                    "upper_square": square_dev_upper,
                }
            )

        # Generate LaTeX table
        latex = []
        latex.append(r"    \begin{tabular}{lcc}")
        latex.append(r"        \toprule")
        latex.append(r"        Method & Mean Deviation & Mean Squared Deviation \\")
        latex.append(r"        \midrule")
        for res in results:
            name = res["name"]
            dev_mean = res["mean_dev_metric"]
            dev_lower = res["dev_lower"]
            dev_upper = res["dev_upper"]

            square_mean = res["mean_square_metric"]
            square_lower = res["lower_square"]
            square_upper = res["upper_square"]

            dev_str = (
                f"${dev_mean:.4f}"
                + "_{-"
                + f"{dev_mean - dev_lower:.4f}"
                + "}"
                + "^{+"
                + f"{dev_upper - dev_mean:.4f}"
                + "}$"
            )
            square_str = (
                f"${square_mean:.4f}"
                + "_{-"
                + f"{square_mean - square_lower:.4f}"
                + "}"
                + "^{+"
                + f"{square_upper - square_mean:.4f}"
                + "}$"
            )

            latex.append(f"        {name} & {dev_str} & {square_str} \\\\")
        latex.append(r"        \bottomrule")
        latex.append(r"    \end{tabular}")
        latex_str = "\n".join(latex)
        file_name = f"{variable_key}_reconstruction_metrics_table.tex"
        if save_dir is not None:
            file_name = os.path.join(save_dir, file_name)
        with open(file_name, "w") as f:
            f.write(latex_str)
        return latex_str

    def plot_relative_neutrino_deviations(
        self, bins=20, xlims=None, coords="cartesian"
    ):
        """
        Plot deviation distributions for magnitude and direction of neutrino momenta

        :param bins: Number of bins
        :param xlims: Optional x-axis limits
        """
        true_neutrino = self.y_test["regression"]
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        neutrino_deviations = []
        names = []
        nu_flows = False
        for i, reconstructor in enumerate(self.prediction_manager.reconstructors):
            if reconstructor.use_nu_flows and not nu_flows:
                nu_flows = True
                names.append(r"$\nu^2$-Flows")
            elif reconstructor.use_nu_flows and nu_flows:
                continue
            elif (
                isinstance(reconstructor, GroundTruthReconstructor)
                and not reconstructor.perform_regression
            ):
                continue
            else:
                names.append(reconstructor.get_neutrino_name())

            pred_neutrino = self.prediction_manager.get_neutrino_predictions(i)

            if coords == "spherical":
                true_neutrino_mag = np.linalg.norm(true_neutrino[..., :3], axis=-1)
                pred_neutrino_mag = np.linalg.norm(pred_neutrino[..., :3], axis=-1)

                mag_deviation = (
                    pred_neutrino_mag - true_neutrino_mag
                ) / true_neutrino_mag

                mag_product = true_neutrino_mag * pred_neutrino_mag

                dir_deviation = np.arccos(
                    np.clip(
                        np.divide(
                            np.sum(
                                pred_neutrino[..., :3] * true_neutrino[..., :3], axis=-1
                            ),
                            mag_product,
                            out=np.zeros_like(true_neutrino_mag),
                            where=((mag_product) != 0)
                            & (~np.isnan(mag_product) & ~np.isinf(mag_product)),
                        ),
                        -1.0,
                        1.0,
                    )
                )
                neutrino_deviations.append(np.array([mag_deviation, dir_deviation]))
            elif coords == "cartesian":
                x_deviation = (pred_neutrino[..., 0] - true_neutrino[..., 0]) / 1e3
                y_deviation = (pred_neutrino[..., 1] - true_neutrino[..., 1]) / 1e3
                z_deviation = (pred_neutrino[..., 2] - true_neutrino[..., 2]) / 1e3
                neutrino_deviations.append(
                    np.array([x_deviation, y_deviation, z_deviation])
                )
            elif coords == "spherical_lepton_fixed":
                lepton_inputs = self.X_test["lep_inputs"]
                lepton_3vect = lorentz_vector_from_PtEtaPhiE_array(
                    lepton_inputs[..., :4]
                )[..., :3]

                true_neutrino_z = project_vectors_onto_axis(
                    true_neutrino[..., :3], lepton_3vect
                )
                pred_neutrino_z = project_vectors_onto_axis(
                    pred_neutrino[..., :3], lepton_3vect
                )

                z_deviation = (pred_neutrino_z - true_neutrino_z) / 1e3
                true_neutrino_perp = true_neutrino[..., :3] - np.expand_dims(
                    true_neutrino_z, axis=-1
                ) * (
                    lepton_3vect / np.linalg.norm(lepton_3vect, axis=-1, keepdims=True)
                )
                pred_neutrino_perp = pred_neutrino[..., :3] - np.expand_dims(
                    pred_neutrino_z, axis=-1
                ) * (
                    lepton_3vect / np.linalg.norm(lepton_3vect, axis=-1, keepdims=True)
                )
                true_neutrino_perp_mag = np.linalg.norm(true_neutrino_perp, axis=-1)
                pred_neutrino_perp_mag = np.linalg.norm(pred_neutrino_perp, axis=-1)
                perp_mag_deviation = np.divide(
                    pred_neutrino_perp_mag - true_neutrino_perp_mag,
                    true_neutrino_perp_mag,
                    out=np.zeros_like(true_neutrino_perp_mag),
                    where=true_neutrino_perp_mag != 0,
                )
                perp_dot_product = np.sum(
                    true_neutrino_perp * pred_neutrino_perp, axis=-1
                )
                perp_mag_product = true_neutrino_perp_mag * pred_neutrino_perp_mag
                perp_angle_deviation = np.arccos(
                    np.clip(
                        np.divide(
                            perp_dot_product,
                            perp_mag_product,
                            out=np.zeros_like(true_neutrino_perp_mag),
                            where=(perp_mag_product != 0)
                            & (
                                ~np.isnan(perp_mag_product)
                                & ~np.isinf(perp_mag_product)
                            ),
                        ),
                        -1.0,
                        1.0,
                    )
                )
                neutrino_deviations.append(
                    np.array([perp_mag_deviation, perp_angle_deviation, z_deviation])
                )

        if coords == "spherical":
            component_labels = [
                r"$\Delta |\vec{p}| / |\vec{p}_{\text{true}}|$",
                r"$\Delta \phi$",
            ]
        elif coords == "spherical_lepton_fixed":
            component_labels = [
                r"$\Delta |\vec{p}_{\perp}| / |\vec{p}_{\perp, \text{true}}|$",
                r"$\Delta \phi_{\perp}$",
                r"$\Delta p_{z}$ [GeV]",
            ]
        elif coords == "cartesian":
            component_labels = [
                r"$\Delta p_x$ [GeV]",
                r"$\Delta p_y$ [GeV]",
                r"$\Delta p_z$ [GeV]",
            ]

        fig, ax = plt.subplots(
            figsize=(6 * len(component_labels), 5 * self.config.NUM_LEPTONS),
            ncols=len(component_labels),
            nrows=self.config.NUM_LEPTONS,
            constrained_layout=True,
        )

        for lepton_idx in range(self.config.NUM_LEPTONS):
            for comp_idx, component in enumerate(component_labels):
                ax_i = ax[lepton_idx, comp_idx]
                DistributionPlotter.plot_feature_distributions(
                    [
                        neutrino_deviations[i][comp_idx][..., lepton_idx]
                        for i in range(len(neutrino_deviations))
                    ],
                    f"{component}" + r"$(\nu_{" + f"{lepton_idx+1}" + r"})$",
                    event_weights=event_weights,
                    labels=names,
                    bins=bins,
                    config=(
                        PlotConfig(xlims=xlims) if xlims is not None else PlotConfig()
                    ),
                    ax=ax_i,
                )

        # Collect handles and labels from the first axis
        handles, labels = ax[0, 0].get_legend_handles_labels()

        # Remove individual legends from all axes
        for lepton_idx in range(self.config.NUM_LEPTONS):
            for comp_idx in range(len(component_labels)):
                leg = ax[lepton_idx, comp_idx].get_legend()
                if leg is not None:
                    leg.remove()
                # pass

        # Add single legend for the whole figure
        fig.legend(
            handles,
            labels,
            loc="center",
            # bbox_to_anchor=(0.5, 1.04),
            ncol=len(names),
        )

        return fig, ax

    def plot_all_deviations(self, save_dir: Optional[str] = None):
        """
        Plot all deviation distributions for all variables and reconstructors.

        Args:
            save_dir: Optional directory to save the plots
        """
        for variable_key in self.variable_configs.keys():
            config = self.variable_configs[variable_key]
            deviation_config = config.get("resolution", {})
            use_relative_deviation = deviation_config.get(
                "use_relative_deviation", False
            )
            deviation_function = deviation_config.get("deviation_function", None)
            variable_label = config["label"]
            if deviation_config.get("deviation_label"):
                variable_label = deviation_config["deviation_label"]

            fig, axes = self.plot_deviations_distributions_all_reconstructors(
                variable_name=variable_key,
                variable_label=variable_label,
                use_relative_deviation=use_relative_deviation,
                deviation_function=deviation_function,
            )
            if save_dir is not None:
                file_name = f"{variable_key}_deviation_distributions.pdf"
                file_path = os.path.join(save_dir, file_name)
                fig.savefig(file_path)
            plt.close(fig)
            self.save_reconstruction_variable_latex_table(
                variable_key=variable_key,
                n_bootstrap=10,
                confidence=0.95,
                save_dir=save_dir,
                use_signed_deviation=deviation_config.get("use_signed_deviation", True),
                use_relative_deviation=use_relative_deviation,
                deviation_function=deviation_function,
            )

    def plot_all_confusion_matrices(self, save_dir: Optional[str] = None, **kwargs):
        """
        Plot confusion matrices for all variables and reconstructors.

        Args:
            save_dir: Optional directory to save the plots
            **kwargs: Additional arguments passed to plot_variable_confusion_matrix_for_all_reconstructors
        """
        for variable_key in self.variable_configs.keys():
            config = self.variable_configs[variable_key]
            variable_label = config["label"]

            figs = self.plot_variable_confusion_matrix_for_all_reconstructors(
                variable_name=variable_key,
                variable_label=variable_label,
                **kwargs,
            )
            if save_dir is not None:
                for reco_name in figs:
                    fig, ax = figs[reco_name]
                    file_name = f"{variable_key}_{convert_reco_name(reco_name)}_confusion_matrix.pdf"
                    file_path = os.path.join(save_dir, file_name)
                    fig.savefig(file_path)
                    plt.close(fig)

    def plot_accuracy_evaluation(self, save_dir: Optional[str] = None, **kwargs):
        """
        Evaluate and print accuracy for all reconstructors.

        Args:
            save_dir: Optional directory to save the results
            **kwargs: Additional arguments for evaluation
        """

        self.save_accuracy_latex_table(save_dir=os.path.join(save_dir), **kwargs)
        fig, ax = self.plot_all_accuracies(**kwargs)
        if save_dir is not None:
            file_name = f"assignment_accuracies.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)
        fig, ax = self.plot_all_selection_accuracies(**kwargs)
        if save_dir is not None:
            file_name = f"selection_accuracies.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)

    def plot_binned_performance_evaluation(
        self,
        save_dir: Optional[str] = None,
        feature_type: str = "jet",
        feature_name: str = "pt",
        fancy_feature_label: Optional[str] = None,
        rescale_factor: Optional[float] = None,
        center_bins: bool = False,
        accuracy_only=False,
        **kwargs,
    ):
        """
        Evaluate and plot binned performance metrics for all reconstructors.

        Args:
            save_dir: Optional directory to save the results
            feature_data_type: Type of feature data for binning
            feature_name: Name of feature for binning
            fancy_feature_label: Optional fancy label for the feature
            **kwargs: Additional arguments for plotting

        """
        fig, ax = self.plot_binned_accuracy(
            feature_data_type=feature_type,
            feature_name=feature_name,
            fancy_feature_label=fancy_feature_label,
            **kwargs,
        )
        if rescale_factor is not None:
            scale_axis_tick_labels(ax, rescale_factor)
        if center_bins:
            center_axis_ticks(ax)
        if save_dir is not None:
            file_name = f"binned_accuracies_{feature_name}.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)
        fig, ax = self.plot_binned_accuracy_quotients(
            feature_data_type=feature_type,
            feature_name=feature_name,
            fancy_feature_label=fancy_feature_label,
            **kwargs,
        )
        if rescale_factor is not None:
            scale_axis_tick_labels(ax, rescale_factor)
        if center_bins:
            center_axis_ticks(ax)
        if save_dir is not None:
            file_name = f"binned_accuracy_quotients_{feature_name}.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)
        fig, ax = self.plot_binned_selection_accuracy(
            feature_data_type=feature_type,
            feature_name=feature_name,
            fancy_feature_label=fancy_feature_label,
            **kwargs,
        )
        if rescale_factor is not None:
            scale_axis_tick_labels(ax, rescale_factor)
        if center_bins:
            center_axis_ticks(ax)
        if save_dir is not None:
            file_name = f"binned_selection_accuracies_{feature_name}.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)
        if accuracy_only:
            return

        for variable_key in self.variable_configs.keys():

            fig, ax = self.plot_binned_variable(
                variable_key=variable_key,
                metric_type="deviation",
                feature_data_type=feature_type,
                feature_name=feature_name,
                fancy_feature_label=fancy_feature_label,
                **kwargs,
            )
            if rescale_factor is not None:
                scale_axis_tick_labels(ax, rescale_factor)
            if center_bins:
                center_axis_ticks(ax)
            if save_dir is not None:
                file_name = f"binned_deviation_{variable_key}_{feature_name}.pdf"
                file_path = os.path.join(save_dir, file_name)
                fig.savefig(file_path)
            plt.close(fig)

    def plot_2d_binned_performance_evaluation(
        self,
        feature1_config: BinningVariableConfig,
        feature2_config: BinningVariableConfig,
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Evaluate and plot 2D binned performance metrics for all reconstructors.

        Args:
            feature1_config: Configuration for the first feature (for x-axis)
            feature2_config: Configuration for the second feature (for y-axis)
            save_dir: Optional directory to save the results
            **kwargs: Additional arguments for plotting
        """
        fig, ax = self.plot_2d_binned_accuracy(
            feature1_binning_config=feature1_config,
            feature2_binning_config=feature2_config,
            **kwargs,
        )
        if save_dir is not None:
            file_name = f"2d_binned_accuracies_{feature1_config.feature_name}_{feature2_config.feature_name}.pdf"
            file_path = os.path.join(save_dir, file_name)
            fig.savefig(file_path)
        plt.close(fig)

    def plot_neutrino_deviation_evaluation(
        self, save_dir: Optional[str] = None, **kwargs
    ):
        """
        Evaluate and plot neutrino deviation metrics for all reconstructors.

        Args:
            save_dir: Optional directory to save the results
            **kwargs: Additional arguments for plotting

        """
        for coords in ["cartesian", "spherical", "spherical_lepton_fixed"]:
            fig, ax = self.plot_relative_neutrino_deviations(coords=coords, **kwargs)
            if save_dir is not None:
                file_name = f"neutrino_deviation_distributions_{coords}.pdf"
                file_path = os.path.join(save_dir, file_name)
                fig.savefig(file_path)
            plt.close(fig)
        self.save_regression_error_latex_table(
            save_dir=os.path.join(save_dir), **kwargs
        )

    def plot_all_distributions(self, save_dir: Optional[str] = None, **kwargs):
        """
        Plot all distributions for all variables and reconstructors.

        Args:
            save_dir: Optional directory to save the plots
            **kwargs: Additional arguments for plotting
        """
        for variable_key in self.variable_configs.keys():
            config = self.variable_configs[variable_key]
            variable_label = config["label"]
            kwargs["xlims"] = config.get("xlims", None)
            kwargs["bins"] = config.get("bins", 20)
            fig, axes = self.plot_distributions_all_reconstructors(
                variable_name=variable_key,
                variable_label=variable_label,
                **kwargs,
            )
            if save_dir is not None:
                file_name = f"{variable_key}_distributions.pdf"
                file_path = os.path.join(save_dir, file_name)
                fig.savefig(file_path)
            plt.close(fig)
