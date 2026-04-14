"""Plotting utilities for evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl

ampl.use_atlas_style()
import seaborn as sns
from typing import Tuple, Optional, List

from .evaluator_utils import (
    PlotConfig,
    BinningUtility,
    BootstrapCalculator,
    FeatureExtractor,
)


class BarPlotter:
    """Handles plotting of bar charts for overall metrics."""

    @staticmethod
    def plot_bar_chart(
        reconstructor_names: List[str],
        metric_values: List[Tuple[float, float, float]],
        metric_label: str,
        config: PlotConfig = PlotConfig(),
    ):
        """
        Plot a bar chart with error bars for overall metrics.

        Args:
            reconstructor_names: List of reconstructor names
            metric_values: List of (mean, lower, upper) tuples for the metric
            metric_label: Label for the y-axis
            config: Plot configuration

        Returns:
            Tuple of (figure, axis)
        """
        names = reconstructor_names
        means = [val[0] for val in metric_values]
        lowers = [val[1] for val in metric_values]
        uppers = [val[2] for val in metric_values]

        errors_lower = [(mean - lower) for mean, lower in zip(means, lowers)]
        errors_upper = [(upper - mean) for mean, upper in zip(means, uppers)]

        fig, ax = plt.subplots(figsize=config.figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            means,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ampl.set_ylabel(metric_label, ax=ax)
        ax.set_title(f"{metric_label} Comparison ({config.confidence*100:.0f}% CI)")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=config.alpha)
        ampl.draw_atlas_label(
            x=0.02, y=0.98, ax=ax, status="Simulation Work in Progress"
        )

        return fig, ax


class BinnedFeaturePlotter:
    """Handles plotting of binned feature metrics."""

    @staticmethod
    def _add_count_histogram(ax, bin_centers, bin_counts, bins):
        """Add event count histogram to plot."""
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            # label="Event Count",
        )
        ampl.set_ylabel("Event Count (a. u.)", color="red", ax=ax_twin)
        ax_twin.tick_params(axis="y", labelcolor="red")

    @staticmethod
    def plot_binned_feature(
        bin_centers: np.ndarray,
        binned_values: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        value_label: str,
        config: PlotConfig = PlotConfig(),
    ):
        """Plot binned feature vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")
        fmt_map = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8"]

        # Plot each reconstructor
        for index, (name, (mean_val, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_values)
        ):
            if config.show_errorbar:
                errors_lower = mean_val - lower
                errors_upper = upper - mean_val
                ax.errorbar(
                    bin_centers,
                    mean_val,
                    yerr=[errors_lower, errors_upper],
                    fmt=fmt_map[index % len(fmt_map)],
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_val,
                    label=name,
                    color=color_map(index),
                )

        # Configure main axes
        ampl.set_xlabel(feature_label, ax=ax)
        ampl.set_ylabel(value_label, ax=ax)
        ax.set_xlim(bins[0], bins[-1])
        if config.xlims is not None:
            ax.set_xlim(config.xlims)
        if config.ylims is not None:
            ax.set_ylim(config.ylims)
        ax.grid(alpha=config.alpha)
        # ampl.draw_legend(ax=ax)
        ax.legend(loc=config.legend_loc)
        ampl.draw_atlas_label(
            x=0.02, y=0.98, ax=ax, status="Simulation Work in Progress"
        )

        # Add event count histogram
        BinnedFeaturePlotter._add_count_histogram(ax, bin_centers, bin_counts, bins)
        return fig, ax

    @staticmethod
    def plot_2d_binned_feature(
        binned_values: List[np.ndarray],
        reconstructor_names: List[str],
        bins_x: np.ndarray,
        bins_y: np.ndarray,
        feature_label_x: str,
        feature_label_y: str,
        value_label: str,
        config: PlotConfig = PlotConfig(),
    ):
        """Plot 2D binned feature vs. two features."""
        num_reconstructors = len(reconstructor_names)
        num_cols = int(np.ceil(np.sqrt(num_reconstructors)))
        num_rows = int(np.ceil(num_reconstructors / num_cols))
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(config.figsize[0] * num_cols, config.figsize[1] * num_rows),
        )
        axes = axes.flatten() if num_reconstructors > 1 else [axes]

        value_min = min(np.nanmin(val) for val in binned_values)
        value_max = max(np.nanmax(val) for val in binned_values)

        color_map = plt.get_cmap("viridis")

        for index, (name, mean_val) in enumerate(
            zip(reconstructor_names, binned_values)
        ):
            ax = axes[index]
            im = ax.imshow(
                mean_val.T,
                origin="lower",
                cmap="viridis",
                extent=[
                    bins_x[0],
                    bins_x[-1],
                    bins_y[0],
                    bins_y[-1],
                ],
                aspect="auto",
                vmin=value_min,
                vmax=value_max,
            )
            ampl.set_xlabel(feature_label_x, ax=ax)
            ampl.set_ylabel(feature_label_y, ax=ax)
            ax.set_title(name)
            ampl.draw_atlas_label(
                x=0.02, y=0.98, ax=ax, status="Simulation Work in Progress"
            )

        # Add combined colorbar
        fig.colorbar(im, ax=axes, label=value_label, shrink=0.6)

        # Remove unused subplots
        for j in range(index + 1, len(axes)):
            fig.delaxes(axes[j])
        return fig, axes[: index + 1]


class ConfusionMatrixPlotter:
    """Handles plotting of confusion matrices."""

    @staticmethod
    def plot_confusion_matrices(
        true_labels: np.ndarray,
        predictions_list: List[np.ndarray],
        reconstructor_names: List[str],
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5),
    ):
        """
        Plot confusion matrices for all reconstructors.

        Args:
            true_labels: True assignment labels
            predictions_list: List of prediction arrays
            reconstructor_names: List of reconstructor names
            normalize: Whether to normalize the confusion matrix
            figsize_per_plot: Size of each subplot

        Returns:
            Tuple of (figure, axes)
        """
        from sklearn.metrics import confusion_matrix

        n_reconstructors = len(reconstructor_names)
        rows = int(np.ceil(np.sqrt(n_reconstructors)))
        cols = int(np.ceil(n_reconstructors / rows))

        fig, axes = plt.subplots(
            cols,
            rows,
            figsize=(figsize_per_plot[0] * rows, figsize_per_plot[1] * cols),
        )
        axes = axes.flatten() if n_reconstructors > 1 else [axes]

        true_indices = np.argmax(true_labels, axis=-2).flatten()

        for i, (name, predictions) in enumerate(
            zip(reconstructor_names, predictions_list)
        ):
            predicted_indices = np.argmax(predictions, axis=-2).flatten()

            cm = confusion_matrix(
                true_indices,
                predicted_indices,
                normalize="true" if normalize else None,
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                ax=axes[i],
                cmap="Blues",
                cbar_kws={"label": "Normalized Count" if normalize else "Count"},
            )
            ampl.draw_tag(tag=name, ax=axes[i])
            ampl.set_xlabel("Predicted Label", ax=axes[i])
            ampl.set_ylabel("True Label", ax=axes[i])
            ampl.draw_atlas_label(
                x=0.02, y=0.98, ax=axes[i], status="Simulation Work in Progress"
            )

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        # fig.suptitle("Confusion Matrix")

        return fig, axes[: i + 1]

    @staticmethod
    def plot_variable_confusion_matrix(
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        variable_label: str,
        axes: plt.Axes,
        bins: np.ndarray,
        normalize: Optional[str] = None,
        plot_mean=False,
        **kwargs,
    ):
        """Plot confusion matrix for a specific variable."""

        hist, xedges, yedges = np.histogram2d(
            true_values, predicted_values, bins=[bins, bins]
        )

        if normalize == "true":
            hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-6)
        elif normalize == "pred":
            hist = hist / (hist.sum(axis=0, keepdims=True) + 1e-6)
        elif normalize == "all":
            hist = hist / (hist.sum() + 1e-6)
        else:
            pass  # No normalization

        mesh, cbar = ampl.plot.plot_2d(
            xedges,
            yedges,
            hist.T,
            ax=axes,
            **kwargs,
        )

        if plot_mean:
            bin_centers_x = 0.5 * (xedges[:-1] + xedges[1:])
            avg_y = []
            for i in range(len(xedges) - 1):
                mask = (true_values.flatten() >= xedges[i]) & (
                    true_values.flatten() < xedges[i + 1]
                )
                if np.sum(mask) > 0:
                    avg_y.append(np.mean(predicted_values.flatten()[mask]))
                else:
                    avg_y.append(np.nan)
            axes.plot(
                bin_centers_x,
                avg_y,
                color="red",
                marker="o",
                linestyle="--",
                label=f"Mean Prediction",
            )

        ampl.set_xlabel(f"True {variable_label}", ax=axes)
        ampl.set_ylabel(f"Reco {variable_label}", ax=axes)
        ampl.draw_atlas_label(
            x=0.02, y=0.98, ax=axes, status="Simulation Work in Progress"
        )
        ampl.draw_legend(ax=axes)
        # axes.get_figure().colorbar(mesh, ax=axes, label="Normalized Count" if normalize else "Count")
        return axes


class ResolutionPlotter:
    """Handles plotting of mass resolution metrics."""

    @staticmethod
    def plot_binned_resolution(
        bin_centers: np.ndarray,
        binned_resolutions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        reconstructor_names: List[str],
        bin_counts: np.ndarray,
        bins: np.ndarray,
        feature_label: str,
        resolution_label: str,
        config: PlotConfig = PlotConfig(),
    ):
        """Plot binned resolution vs. a feature."""
        fig, ax = plt.subplots(figsize=config.figsize)
        color_map = plt.get_cmap("tab10")
        fmt_map = ["o", "s", "D", "^", "v", "P", "*", "X", "h", "8"]
        # Plot each reconstructor
        for index, (name, (mean_res, lower, upper)) in enumerate(
            zip(reconstructor_names, binned_resolutions)
        ):

            if config.show_errorbar:
                errors_lower = mean_res - lower
                errors_upper = upper - mean_res
                ax.errorbar(
                    bin_centers,
                    mean_res,
                    yerr=[errors_lower, errors_upper],
                    fmt=fmt_map[index % len(fmt_map)],
                    label=name,
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    mean_res,
                    label=name,
                    color=color_map(index),
                )

        # Configure axes
        ampl.set_xlabel(feature_label, ax=ax)
        ampl.set_ylabel(resolution_label, ax=ax)
        ampl.draw_atlas_label(
            x=0.02, y=0.98, ax=ax, status="Simulation Work in Progress"
        )

        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=config.alpha)
        ax.legend(loc=config.legend_loc)

        # Add event count histogram
        ax_twin = ax.twinx()
        ax_twin.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_twin.set_ylabel("Event Count", color="red")
        ax_twin.tick_params(axis="y", labelcolor="red")

        # Set title
        title = f"{resolution_label} per Bin vs {feature_label}"
        if config.show_errorbar:
            title += f" ({config.confidence*100:.0f}% CI)"
        # ax.set_title(title)

        return fig, ax


class DistributionPlotter:
    """Handles plotting of general distributions."""

    @staticmethod
    def plot_feature_distributions(
        feature_values: np.ndarray,
        feature_label: str,
        event_weights: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        ax: Optional[plt.Axes] = None,
        bins: int = 50,
        config: PlotConfig = PlotConfig(),
    ):
        """
        Plot histogram of a feature's distribution.

        Args:
            feature_values: Array of feature values (n_events,)
            feature_name: Name of the feature
            feature_label: Label for the x-axis
            event_weights: Optional event weights (n_events,)
            bins: Number of bins for histogram
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if event_weights is None:
            event_weights = np.ones(len(feature_values[0]))

        # Determine bins
        if config is not None and config.xlims is not None:
            bin_edges = np.linspace(config.xlims[0], config.xlims[1], bins + 1)
        else:
            # Combine all data to get consistent binning
            all_values = np.concatenate(feature_values)
            bin_edges = np.linspace(
                np.percentile(all_values, 1), np.percentile(all_values, 99), bins + 1
            )

        # Plot histograms
        for idx, values in enumerate(feature_values):
            label = labels[idx] if labels is not None else None

            # Filter out NaN and inf values
            valid_mask = np.isfinite(values)
            valid_values = values[valid_mask]
            valid_weights = event_weights[valid_mask]

            if len(valid_values) == 0:
                print(f"Warning: No valid values for {label}")
                continue

            ax.hist(
                valid_values,
                bins=bin_edges,
                weights=valid_weights,
                # alpha=0.5,
                label=label,
                histtype="step",
                linewidth=2,
                density=True,
                color=plt.get_cmap("tab10")(idx),
            )
        ampl.set_xlabel(feature_label, ax=ax)
        ampl.set_ylabel("Density", ax=ax)
        ampl.draw_atlas_label(
            x=0.02, y=0.98, ax=ax, status="Simulation Work in Progress"
        )
        if np.isfinite(bin_edges).all() and not np.isnan(bin_edges).any():
            ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.grid(alpha=0.3)
        if labels is not None:
            ax.legend(loc=config.legend_loc)
        return ax


def convert_reco_name(string: str) -> str:
    """Convert printable latex-like reconstructor name to a file-name."""
    return (
        string.replace(" ", "_")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("$", "")
        .replace("\\", "")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )
