"""Evaluator for ML-based jet assignment models."""

import numpy as np
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl

ampl.use_atlas_style()
from typing import Optional
from copy import deepcopy

from .evaluator_utils import (
    BootstrapCalculator,
    AccuracyCalculator,
    SelectionAccuracyCalculator,
    NeutrinoDeviationCalculator,
)
from ..reconstruction import KerasFFRecoBase
from typing import List, Union
import time
import os


class FeatureImportanceCalculator:
    """Calculate feature importance using permutation importance."""

    def __init__(
        self,
        reconstructor: KerasFFRecoBase,
        X_test: dict,
        y_test: dict,
    ):
        self.reconstructor = reconstructor
        # Filter X_test to only include features used by this reconstructor
        self.X_test = X_test
        self.y_test = y_test
        self.config = reconstructor.config
        self.padding_value = self.config.padding_value

    def compute_permutation_importance(
        self, num_repeats: int = 5, evaluate_regression=False
    ) -> dict:
        """
        Compute feature importance using permutation importance.

        Args:
            num_repeats: Number of times to permute each feature

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before computing "
                "permutation importance."
            )

        assignment_baseline_prediction, regression_baseline_prediction = (
            self.reconstructor.complete_forward_pass(self.X_test)
        )

        assignment_baseline_performance = AccuracyCalculator.compute_accuracy(
            assignment_baseline_prediction,
            self.y_test["assignment"],
            per_event=False,
        )
        print(f"Baseline Assignment Performance: {assignment_baseline_performance:.4f}")

        regression_baseline_performance = None
        if evaluate_regression:
            regression_baseline_performance = (
                NeutrinoDeviationCalculator.compute_relative_deviation(
                    regression_baseline_prediction,
                    self.y_test["regression"],
                )
            )
            print(
                f"Baseline Regression Performance: {regression_baseline_performance:.4f}"
            )

        assingment_importances = {}
        regression_importances = {}

        # Compute importance for each available feature type
        available_feature_types = [
            "jet_inputs",
            "lep_inputs",
            "met_inputs",
            "global_event_inputs",
        ]

        for feature_type in available_feature_types:
            # Check if this feature type is available in both config and test data
            if (
                feature_type not in self.config.feature_indices
                or feature_type not in self.X_test
            ):
                continue

            assignment_type_importances, regression_type_importances = (
                self._compute_feature_type_importance(
                    feature_type,
                    num_repeats,
                    assignment_baseline_performance,
                    regression_baseline_performance,
                )
            )

            assingment_importances.update(assignment_type_importances)
            if evaluate_regression:
                regression_importances.update(regression_type_importances)

        return (
            assingment_importances
            if not evaluate_regression
            else (assingment_importances, regression_importances)
        )

    def _compute_feature_type_importance(
        self,
        feature_type: str,
        num_repeats: int,
        assignment_baseline_performance: float,
        regression_baseline_performance: Optional[float] = None,
    ) -> dict:
        """Compute importance for a specific feature type."""
        assignment_importances = {}
        regression_importances = {}
        feature_indices = self.config.feature_indices[feature_type]

        for feature_name, feature_idx in feature_indices.items():
            assignment_scores = []
            regression_scores = []

            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)

                if feature_type == "jet_inputs":
                    # Only permute non-padded jet features
                    mask = np.any(
                        X_permuted["jet_inputs"] != self.padding_value, axis=-1
                    )
                    X_permuted["jet_inputs"][mask, feature_idx] = np.random.permutation(
                        X_permuted["jet_inputs"][mask, feature_idx]
                    )
                elif X_permuted[feature_type].ndim == 3:
                    # Permute lepton/MET features directly
                    X_permuted[feature_type][:, :, feature_idx] = np.random.permutation(
                        X_permuted[feature_type][:, :, feature_idx]
                    )
                else:
                    # Permute global event features directly
                    X_permuted[feature_type][:, feature_idx] = np.random.permutation(
                        X_permuted[feature_type][:, feature_idx]
                    )

                permutated_assignment_pred, permutated_regression_pred = (
                    self.reconstructor.complete_forward_pass(X_permuted)
                )
                assignment_performance = -(
                    AccuracyCalculator.compute_accuracy(
                        permutated_assignment_pred,
                        self.y_test["assignment"],
                        per_event=False,
                    )
                    - assignment_baseline_performance
                )
                assignment_scores.append(assignment_performance)
                if regression_baseline_performance is not None:
                    regression_performance = -(
                        NeutrinoDeviationCalculator.compute_relative_deviation(
                            permutated_regression_pred,
                            self.y_test["nu_flows_neutrino_truth"],
                        )
                        - regression_baseline_performance
                    )
                    regression_scores.append(regression_performance)

            # Store mean importance across repeats
            assignment_importances[feature_name] = np.mean(assignment_scores)
            if regression_baseline_performance is not None:
                regression_importances[feature_name] = np.mean(regression_scores)

        return assignment_importances, regression_importances


class MLEvaluator:
    """Evaluator for ML-based jet assignment models."""

    def __init__(
        self,
        reconstructor: Union[KerasFFRecoBase, List[KerasFFRecoBase]],
        X_test: Union[dict, List[dict]],
        y_test: Union[dict, List[dict]],
    ):
        # Handle both single reconstructor and list of reconstructors
        self.reconstructors = (
            reconstructor if isinstance(reconstructor, list) else [reconstructor]
        )
        self.X_test = (
            X_test if isinstance(X_test, list) else [X_test] * len(self.reconstructors)
        )
        self.y_test = (
            y_test if isinstance(y_test, list) else [y_test] * len(self.reconstructors)
        )

        # Store configuration from first reconstructor (for backward compatibility)
        first_reconstructor = self.reconstructors[0].config
        self.padding_value = first_reconstructor.padding_value
        self.feature_indices = first_reconstructor.feature_indices

        # Initialize helper classes for each reconstructor (with filtered features)
        self.feature_importance_calcs = [
            FeatureImportanceCalculator(
                reconstructor=rec,
                X_test=self.X_test[idx],
                y_test=self.y_test[idx],
            )
            for idx, rec in enumerate(self.reconstructors)
        ]

    def plot_training_history(self):
        """Plot training and validation loss/accuracy over epochs for all models."""
        for reconstructor in self.reconstructors:
            if reconstructor.history is None:
                if not reconstructor.perform_regression:
                    model_name = reconstructor.get_assignment_name()
                else:
                    model_name = reconstructor.get_full_reco_name()
                raise ValueError(
                    f"No training history found for model {model_name}. "
                    "Please train the model before plotting history."
                )

        # Adjust subplot layout based on whether regression metrics exist

        color_map = plt.get_cmap("tab10")

        unique_metrics = set()
        for reconstructor in self.reconstructors:
            unique_metrics.update(
                [
                    metric.replace("val_", "")
                    for metric in reconstructor.history.history.keys()
                    if metric.startswith("val_")
                ]
            )
        unique_metrics = list(unique_metrics)
        num_metrics = len(unique_metrics)
        num_cols = int(np.ceil(np.sqrt(num_metrics)))
        num_rows = int(np.ceil(num_metrics / num_cols))
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows)
        )
        axes = axes.flatten()
        for idx, metric_name in enumerate(unique_metrics):
            ax = axes[idx]
            for rec_idx, reconstructor in enumerate(self.reconstructors):
                history = reconstructor.history
                color = color_map(rec_idx)
                label_prefix = (
                    reconstructor.get_assignment_name()
                    if not reconstructor.perform_regression
                    else reconstructor.get_full_reco_name()
                )
                max_epochs = 0
                if metric_name in history.history:
                    ax.plot(
                        history.history[metric_name],
                        label=f"{label_prefix} Training",
                        color=color,
                        linestyle="-",
                    )
                    if len(history.history[metric_name]) > max_epochs:
                        max_epochs = len(history.history[metric_name])
                if f"val_{metric_name}" in history.history:
                    ax.plot(
                        history.history[f"val_{metric_name}"],
                        label=f"{label_prefix} Validation",
                        color=color,
                        linestyle="--",
                    )
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ampl.draw_atlas_label(
                x=0.02,
                y=0.92,
                ax=ax,
                label="Simulation Work in Progress",
            )
            ampl.set_xlabel(ax=ax, label="Training Epoch")
            ampl.set_ylabel(ax=ax, label=metric_name.replace("_", " ").title())
            ax.set_xlim(-int(max_epochs / 10), max_epochs + int(max_epochs / 10))
            ax.legend()
            ax.grid(True, alpha=0.3)
        # Remove any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        return fig, ax

    @staticmethod
    def _plot_metric(ax, history, metric_name: str, title: str, ylabel: str):
        """Helper method to plot a single training metric."""
        ax.plot(history.history[metric_name], label=f"Training {ylabel}")
        ax.plot(history.history[f"val_{metric_name}"], label=f"Validation {ylabel}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    def compute_permutation_importance(
        self, num_repeats: int = 5, reconstructor_idx: int = 0
    ) -> dict:
        """
        Compute feature importance using permutation importance.

        Args:
            num_repeats: Number of times to permute each feature
            reconstructor_idx: Index of reconstructor to evaluate (default: 0)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance_calcs[
            reconstructor_idx
        ].compute_permutation_importance(num_repeats)

    def plot_feature_importance(
        self,
        num_repeats: int = 10,
        save_dir: Optional[str] = None,
        rename_features: Optional[callable] = None,
    ):
        """
        Plot feature importance scores for all models.

        Args:
            num_repeats: Number of times to permute each feature
            save_dir: Directory to save individual plots (optional)

        Returns:
            List of (fig, ax) tuples for each model
        """
        results = []
        if rename_features is not None:
            print("Renaming features:")
            for feature_type in self.feature_indices:
                for feature_name in self.feature_indices[feature_type]:
                    new_name = rename_features(feature_name)
                    print(f"  {feature_name} -> {new_name}")

        for idx, (reconstructor, calc) in enumerate(
            zip(self.reconstructors, self.feature_importance_calcs)
        ):
            if not reconstructor.perform_regression:
                model_name = reconstructor.get_assignment_name()
            else:
                model_name = reconstructor.get_full_reco_name()
            # Get available features for this model
            available_features = []
            for feature_type in [
                "jet_inputs",
                "lep_inputs",
                "met_inputs",
                "global_event_inputs",
            ]:
                if (
                    hasattr(reconstructor.config, "feature_indices")
                    and feature_type in reconstructor.config.feature_indices
                ):
                    available_features.append(feature_type)

            print(
                f"Computing feature importance for {model_name} "
                f"(features: {', '.join(available_features)})..."
            )

            importances = calc.compute_permutation_importance(
                num_repeats=num_repeats,
                evaluate_regression=reconstructor.perform_regression,
            )

            if not reconstructor.perform_regression:
                assignment_importances = importances

                # Sort features by importance
                sorted_items = sorted(
                    assignment_importances.items(), key=lambda x: x[1], reverse=True
                )
                features = [item[0] for item in sorted_items]
                scores = [item[1] for item in sorted_items]
                features = [item[0] for item in sorted_items]
                if rename_features is not None:
                    features = [rename_features(feature) for feature in features]
                scores = [item[1] for item in sorted_items]

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.barh(features, scores, color="skyblue")
                ampl.draw_atlas_label(
                    x=0.02,
                    y=0.92,
                    ax=ax1,
                    label="Simulation Work in Progress",
                )
                ampl.set_xlabel(ax=ax, label="Importance Score")

                # ax.set_title(f"Feature Importance - {model_name}")
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis="x")

                if save_dir:
                    save_path = f"{save_dir}/{model_name}_feature_importance.pdf"
                    fig.savefig(save_path, dpi=300)
                    print(f"Saved feature importance plot to {save_path}")

                results.append((fig, ax))

            else:
                assignment_importances, regression_importances = importances

                # Plot assignment importances
                sorted_items = sorted(
                    assignment_importances.items(), key=lambda x: x[1], reverse=True
                )
                features = [item[0] for item in sorted_items]
                scores = [item[1] for item in sorted_items]
                features = [item[0] for item in sorted_items]
                if rename_features is not None:
                    features = [rename_features(feature) for feature in features]
                scores = [item[1] for item in sorted_items]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                ax1.barh(features, scores, color="skyblue")
                ampl.draw_atlas_label(
                    x=0.02,
                    y=0.92,
                    ax=ax1,
                    label="Simulation Work in Progress",
                )
                ampl.set_xlabel(ax=ax1, label="Importance Score")
                ax1.set_title(f"Assignment Feature Importance - {model_name}")
                ax1.invert_yaxis()
                ax1.grid(True, alpha=0.3, axis="x")

                # Plot regression importances
                sorted_items = sorted(
                    regression_importances.items(), key=lambda x: x[1], reverse=True
                )
                features = [item[0] for item in sorted_items]
                scores = [item[1] for item in sorted_items]
                features = [item[0] for item in sorted_items]
                if rename_features is not None:
                    features = [rename_features(feature) for feature in features]
                scores = [item[1] for item in sorted_items]

                ax2.barh(features, scores, color="salmon")
                ampl.draw_atlas_label(
                    x=0.02,
                    y=0.92,
                    ax=ax1,
                    label="Simulation Work in Progress",
                )
                ampl.set_xlabel(ax=ax2, label="Importance Score")
                ax2.set_title(f"Regression Feature Importance - {model_name}")
                ax2.invert_yaxis()
                ax2.grid(True, alpha=0.3, axis="x")

                if save_dir:
                    save_path = f"{save_dir}/{model_name}_feature_importance.pdf"
                    fig.savefig(save_path, dpi=300)
                    print(f"Saved feature importance plot to {save_path}")

                results.append((fig, (ax1, ax2)))

        return results

    def evaluate_inference_time(
        self,
        num_samples: Optional[int] = None,
        num_warmup: int = 10,
        num_iterations: int = 100,
    ) -> dict:
        """
        Evaluate inference time for all models using the same number of samples.

        Args:
            num_samples: Number of samples to use for inference testing.
                If None, uses the minimum dataset size across all models.
            num_warmup: Number of warmup iterations before timing
            num_iterations: Number of iterations to average over

        Returns:
            Dictionary mapping model names to inference time statistics
        """

        # Determine the number of samples to use
        if num_samples is None:
            # Use minimum dataset size to ensure all models can be tested equally
            min_samples = min(
                (
                    len(X["jet_inputs"])
                    if "jet_inputs" in X
                    else len(next(iter(X.values())))
                )
                for X in self.X_test
            )
            num_samples = min_samples
            print(f"Using {num_samples} samples for inference time evaluation")
        else:
            # Verify all datasets have enough samples
            for idx, X in enumerate(self.X_test):
                dataset_size = (
                    len(X["jet_inputs"])
                    if "jet_inputs" in X
                    else len(next(iter(X.values())))
                )
                if dataset_size < num_samples:
                    raise ValueError(
                        f"Dataset {idx} has only {dataset_size} samples, "
                        f"but {num_samples} were requested"
                    )

        results = {}

        for idx, reconstructor in enumerate(self.reconstructors):
            if not reconstructor.perform_regression:
                model_name = reconstructor.get_assignment_name()
            else:
                model_name = reconstructor.get_full_reco_name()
            print(f"\nEvaluating inference time for {model_name}...")

            # Prepare test subset with exactly num_samples
            X_subset = {}
            for key, value in self.X_test[idx].items():
                X_subset[key] = value[:num_samples]

            # Warmup iterations
            print(f"  Running {num_warmup} warmup iterations...")
            for _ in range(num_warmup):
                _ = reconstructor.complete_forward_pass(X_subset)

            # Timed iterations
            print(f"  Running {num_iterations} timed iterations...")
            times = []
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = reconstructor.complete_forward_pass(X_subset)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            # Calculate statistics
            times = np.array(times)
            results[model_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "median_time": np.median(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "num_samples": num_samples,
                "time_per_sample": np.mean(times) / num_samples,
            }

            print(f"  Mean time: {results[model_name]['mean_time']*1000:.2f} ms")
            print(f"  Std time: {results[model_name]['std_time']*1000:.2f} ms")
            print(
                f"  Time per sample: {results[model_name]['time_per_sample']*1000:.4f} ms"
            )

        return results

    def plot_inference_time_comparison(
        self,
        num_samples: Optional[int] = 100000,
        num_warmup: int = 3,
        num_iterations: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison of inference times across all models.

        Args:
            num_samples: Number of samples to use for inference testing
            num_warmup: Number of warmup iterations
            num_iterations: Number of iterations to average over
            save_path: Path to save the plot (optional)

        Returns:
            Tuple of (fig, axes)
        """
        results = self.evaluate_inference_time(
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )
        num_models = len(results)

        fig, ax2 = plt.subplots(1, 1, figsize=(num_models * 5, 6))

        model_names = list(results.keys())
        mean_times = [results[name]["mean_time"] * 1000 for name in model_names]
        std_times = [results[name]["std_time"] * 1000 for name in model_names]
        time_per_sample = [
            results[name]["time_per_sample"] * 1000 for name in model_names
        ]

        # Plot 2: Time per sample
        ax2.bar(model_names, time_per_sample, color="coral")
        ax2.set_ylabel("Time per Sample (ms)")
        ax2.set_title("Inference Time per Sample")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.tick_params(axis="x", rotation=45)

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"\nInference time comparison plot saved to {save_path}")

        return fig, ax2

    def evaluate_model_parameters(self) -> dict:
        """
        Evaluate the number of parameters for all models.

        Returns:
            Dictionary mapping model names to parameter counts
        """
        results = {}

        for idx, reconstructor in enumerate(self.reconstructors):
            if not reconstructor.perform_regression:
                model_name = reconstructor.get_assignment_name()
            else:
                model_name = (
                    reconstructor.get_full_reco_name()
                )  # Get total and trainable parameters
            total_params = reconstructor.model.count_params()

            # Count trainable parameters
            trainable_params = sum(
                np.prod(w.shape) for w in reconstructor.model.trainable_weights
            )

            non_trainable_params = total_params - trainable_params

            results[model_name] = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
            }

            print(f"\n{model_name}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Non-trainable parameters: {non_trainable_params:,}")

        return results

    def plot_model_parameters_comparison(
        self,
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison of model parameters across all models.

        Args:
            save_path: Path to save the plot (optional)

        Returns:
            Tuple of (fig, ax)
        """
        results = self.evaluate_model_parameters()
        num_models = len(results)
        fig, ax = plt.subplots(1, 1, figsize=(num_models * 3, 6))

        model_names = list(results.keys())
        trainable_params = [
            results[name]["trainable_params"] / 1e6 for name in model_names
        ]
        non_trainable_params = [
            results[name]["non_trainable_params"] / 1e6 for name in model_names
        ]

        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(
            x,
            trainable_params,
            width,
            label="Trainable",
            color="steelblue",
        )
        ax.bar(
            x,
            non_trainable_params,
            width,
            bottom=trainable_params,
            label="Non-trainable",
            color="lightcoral",
        )

        ampl.draw_atlas_label(
            x=0.02,
            y=0.92,
            ax=ax,
            label="Simulation Work in Progress",
        )
        ampl.set_ylabel(ax=ax, label="Number of Parameters (Millions)")
        ax.set_title("Model Parameters Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"\nModel parameters comparison plot saved to {save_path}")

        return fig, ax

    def save_accuracy_latex_table(
        self,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Save reconstruction accuracies as a LaTeX table.

        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence: Confidence level for intervals
            caption: Table caption
            label: Table label
            save_dir: Directory to save the LaTeX file (optional)
        Returns:
            LaTeX table as a string
        """
        results = []
        for idx, reconstructor in enumerate(self.reconstructors):
            if not reconstructor.perform_regression:
                model_name = reconstructor.get_assignment_name()
            else:
                model_name = reconstructor.get_full_reco_name()  # Get predictions
            assignment_pred, _ = reconstructor.complete_forward_pass(self.X_test[idx])

            # Compute accuracy with bootstrap
            acc_mean, acc_lower, acc_upper = BootstrapCalculator.compute_bootstrap_ci(
                data=AccuracyCalculator.compute_accuracy(
                    true_labels=self.y_test[idx]["assignment"],
                    predictions=assignment_pred,
                    per_event=True,
                ),
                n_bootstrap=n_bootstrap,
                confidence=confidence,
            )
            sel_acc_mean, sel_acc_lower, sel_acc_upper = (
                BootstrapCalculator.compute_bootstrap_ci(
                    data=SelectionAccuracyCalculator.compute_selection_accuracy(
                        true_labels=self.y_test[idx]["assignment"],
                        predictions=assignment_pred,
                        per_event=True,
                    ),
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                )
            )
            results.append(
                {
                    "name": model_name,
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
        print(f"LaTeX table saved to {file_name}")
