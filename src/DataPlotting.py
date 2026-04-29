from .preprocessing.training_data_loader import TrainingDataLoader, DataConfig
import numpy as np
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl
from .utils import lorentz_vector_from_neutrino_momenta_array

ampl.use_atlas_style()
ampl.set_color_cycle("ATLAS")
import os


class DataPlotter:
    def __init__(self, data_processor: TrainingDataLoader, plots_dir: str):
        self.data_processor = data_processor
        self.plots_dir = plots_dir
        self.padding_value = data_processor.data_config.padding_value
        self.max_jets = data_processor.data_config.max_jets
        self.NUM_LEPTONS = data_processor.data_config.NUM_LEPTONS
        self.feature_index_dict = data_processor.data_config.feature_indices
        self.event_cuts = np.ones(data_processor.data_length, dtype=bool)
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_feature_distribution(self, feature_type: str, feature_name: str, bins=50):
        """
        Plots the distribution of a specified feature.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to plot.
            bins (int): Number of bins for the histogram.
        """
        feature_data = self.get_feature_data(feature_type, feature_name)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            feature_data.flatten()[feature_data.flatten() != self.padding_value],
            bins=bins,
            alpha=0.7,
            color="blue",
            edgecolor="black",
            density=True,
        )
        ax.set_title(f"Distribution of {feature_name} ({feature_type})")
        ampl.draw_atlas_label(
            x=0.02, y=0.95, status="Simulation Work in Progress", ax=ax
        )
        ampl.set_xlabel(feature_name, ax=ax)
        ampl.set_ylabel("Arbitrary Units", ax=ax)

        return fig, ax

    def add_feature(self, function, name: str):
        """
        Adds a new feature to the data processor using a custom function.

        Args:
            function (function): A function that takes the data processor as input and returns the new feature array.
            name (str): The name of the new feature.
        """
        self.data_processor.add_custom_feature(function, name)

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of all features.
        """
        features_list = []
        feature_names = []

        # Collect all features
        for feature_type in ["jet_inputs", "lep_inputs", "met_inputs"]:
            for feature_name in self.data_processor.feature_index_dict[feature_type]:
                feature_data = self.get_feature_data(feature_type, feature_name)
                if feature_data.shape[-1] > 1:
                    for i in range(feature_data.shape[-1]):
                        features_list.append(feature_data[..., i].flatten())
                        feature_names.append(f"{feature_name}_{i}")
                else:
                    features_list.append(feature_data.flatten())
                    feature_names.append(feature_name)

        # Compute correlation matrix
        correlation_matrix = np.zeros((len(features_list), len(features_list)))
        for i in range(len(features_list)):
            for j in range(len(features_list)):
                if i <= j:
                    valid_mask = (features_list[i] != self.padding_value) & (
                        features_list[j] != self.padding_value
                    )
                    if np.sum(valid_mask) > 1:
                        corr = np.corrcoef(
                            features_list[i][valid_mask], features_list[j][valid_mask]
                        )[0, 1]
                    else:
                        corr = 0
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
        ax.set_title("Feature Correlation Matrix", pad=20)

        fig.savefig(os.path.join(self.plots_dir, "feature_correlation_matrix.pdf"))
        return fig, ax

    def register_data_cut(self, feature_type: str, feature_name: str, cut_function):
        """
        Registers a data cut based on a feature's value range.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to apply the cut on.
            cut_function (function): A function that takes feature data and returns a boolean mask for valid events.
        """
        feature_data = self.get_feature_data(feature_type, feature_name, uncut=True)
        cut_mask = cut_function(feature_data)
        self.event_cuts = self.event_cuts & cut_mask

    def reset_data_cuts(self):
        """
        Resets all registered data cuts.
        """
        self.event_cuts = np.ones(self.data_processor.data_length, dtype=bool)

    def get_feature_data(self, feature_type: str, feature_name: str, uncut=False):
        """
        Retrieves feature data after applying registered data cuts.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to retrieve.

        Returns:
            np.ndarray: The feature data after applying cuts.
        """
        feature_data = self.data_processor.get_feature_data(feature_type, feature_name)
        if uncut:
            return feature_data
        else:
            return feature_data[self.event_cuts]

    def get_all_feature_data(self, feature_type: str):
        """
        Retrieves all feature data of a specified type after applying registered data cuts.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').

        Returns:
            np.ndarray: The feature data after applying cuts.
        """
        feature_data = self.data_processor.get_all_feature_data(feature_type)
        return feature_data[self.event_cuts]

    def plot_relational_jet_lepton_inputs(
        self, feature_function, name="relational_feature", **kwargs
    ):
        """
        Plots the value of a relational feature between jets and leptons.
        Args:
            feature_function (function): A function that takes jet and lepton feature arrays and computes a relational feature.
        """
        region_tag = kwargs.pop("region_tag", None)
        jet_inputs = self.get_all_feature_data("jet_inputs")
        lepton_inputs = self.get_all_feature_data("lep_inputs")
        labels = self.get_all_feature_data("assignment")
        neutrinos = self.get_all_feature_data("nu_flows_neutrino_regression").reshape(
            -1, 2, 3
        )

        lepton_extended = np.repeat(
            lepton_inputs[:, np.newaxis, :, :], self.max_jets, axis=1
        )
        jet_extended = np.repeat(
            jet_inputs[:, :, np.newaxis, :], self.NUM_LEPTONS, axis=2
        )
        neutrino_extended = np.repeat(
            neutrinos[:, np.newaxis, :, :], self.max_jets, axis=1
        )
        relational_feature = feature_function(
            jet_extended.transpose(-1, 1, 2, 0),
            lepton_extended.transpose(-1, 1, 2, 0),
            neutrino_extended.transpose(-1, 1, 2, 0),
        ).transpose(2, 0, 1)

        jet_mask = (jet_inputs[:, :, :] != self.padding_value).any(
            axis=-1, keepdims=True
        )
        matched_relational_features = relational_feature[
            labels == 1 & jet_mask
        ].flatten()
        unmatched_relational_features = relational_feature[
            (labels == 0) & jet_mask
        ].flatten()

        fig, ax = plt.subplots(figsize=(8, 6))
        _, bins = np.histogram(
            np.concatenate(
                (matched_relational_features, unmatched_relational_features)
            ),
            **kwargs,
        )
        ax.hist(
            matched_relational_features,
            bins=bins,
            alpha=0.7,
            label="Correct $b$-jet",
            color="tab:blue",
            density=True,
        )
        ax.hist(
            unmatched_relational_features,
            bins=bins,
            alpha=0.7,
            label="Other jets",
            color="tab:red",
            density=True,
        )
        ampl.draw_atlas_label(
            x=0.02, y=0.95, status="Simulation Work in Progress", ax=ax, desc=region_tag
        )
        ampl.set_xlabel(name, ax=ax)
        ampl.set_ylabel("Arbitrary Units", ax=ax)
        ax.legend()
        return fig, ax

    def plot_2d_feature_distribution(
        self,
        feature_type_x: str,
        feature_name_x: str,
        feature_type_y: str,
        feature_name_y: str,
        normalise="none",
        plot_average=False,
        **kwargs,
    ):
        """
        Plots a 2D histogram of two specified features.
        Args:
            feature_type_x (str): The type of the x-axis feature (e.g., 'jet', 'lepton', 'met').
            feature_name_x (str): The name of the x-axis feature to plot.
            feature_type_y (str): The type of the y-axis feature (e.g., 'jet', 'lepton', 'met').
            feature_name_y (str): The name of the y-axis feature to plot.
            normalise (str): Normalisation method for the histogram ('none', 'x', 'y', 'all').
            plot_average (bool): Whether to plot the average of the y feature in each x bin
            kwargs: Additional keyword arguments for np.hist2d to customise the binning and appearance.
        """
        feature_data_x = self.get_feature_data(feature_type_x, feature_name_x)
        feature_data_y = self.get_feature_data(feature_type_y, feature_name_y)
        hist2d, bins_x, bins_y = np.histogram2d(
            feature_data_x.flatten()[feature_data_x.flatten() != self.padding_value],
            feature_data_y.flatten()[feature_data_y.flatten() != self.padding_value],
            **kwargs,
        )
        # Normalisation
        if normalise == "x":
            hist2d = hist2d / hist2d.sum(axis=1, keepdims=True)
        elif normalise == "y":
            hist2d = hist2d / hist2d.sum(axis=0, keepdims=True)
        elif normalise == "all":
            hist2d = hist2d / hist2d.sum()

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(
            hist2d.T,
            origin="lower",
            aspect="auto",
            extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
            cmap="viridis",
        )
        if plot_average:
            # Get average of the y feature in each x bin
            bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
            avg_y = []
            for i in range(len(bins_x) - 1):
                mask = (
                    (feature_data_x.flatten() >= bins_x[i])
                    & (feature_data_x.flatten() < bins_x[i + 1])
                    & (feature_data_y.flatten() != self.padding_value)
                )
                if np.sum(mask) > 0:
                    avg_y.append(np.mean(feature_data_y.flatten()[mask]))
                else:
                    avg_y.append(np.nan)
            ax.plot(
                bin_centers_x,
                avg_y,
                color="red",
                marker="o",
                linestyle="-",
                label=f"Average {feature_name_y}",
            )
            ax.legend(loc="upper right")
        fig.colorbar(cax, ax=ax, label="Arbitrary Units")
        ampl.draw_atlas_label(
            x=0.02, y=0.95, status="Simulation Work in Progress", ax=ax
        )
        ampl.set_xlabel(feature_name_x, ax=ax)
        ampl.set_ylabel(feature_name_y, ax=ax)
        return fig, ax

    def plot_binned_feature_mean(
        self,
        binning_feature_type: str,
        binning_feature_name: str,
        target_feature_type: str,
        target_feature_name: str,
        bins: int = 10,
        xlims: tuple = None,
    ):
        """
        Plots the mean of a target feature binned by another feature.

        Args:
            binning_feature_type (str): The type of the feature to bin by (e.g., 'jet', 'lepton', 'met').
            binning_feature_name (str): The name of the feature to bin by.
            target_feature_type (str): The type of the target feature (e.g., 'jet', 'lepton', 'met').
            target_feature_name (str): The name of the target feature to compute the mean of.
            bins (int): Number of bins for the binning feature.
            xlims (tuple): Limits for the x-axis (min, max).
        """
        binning_feature_data = self.get_feature_data(
            binning_feature_type, binning_feature_name
        )
        target_feature_data = self.get_feature_data(
            target_feature_type, target_feature_name
        )

        if xlims is None:
            x_min = np.min(
                binning_feature_data[binning_feature_data != self.padding_value]
            )
            x_max = np.max(
                binning_feature_data[binning_feature_data != self.padding_value]
            )
        else:
            x_min, x_max = xlims

        bin_edges = np.linspace(x_min, x_max, bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mean_values = []
        for i in range(bins):
            mask = (
                (binning_feature_data >= bin_edges[i])
                & (binning_feature_data < bin_edges[i + 1])
                & (target_feature_data != self.padding_value)
            )
            if np.sum(mask) > 0:
                mean_values.append(np.mean(target_feature_data[mask]))
            else:
                mean_values.append(np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bin_centers, mean_values, marker="o", linestyle="-")
        return fig, ax
