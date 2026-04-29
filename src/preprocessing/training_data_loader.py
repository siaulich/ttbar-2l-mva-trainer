"""
Data loading and preprocessing module for machine learning tasks involving jets, leptons, and MET.

This module provides classes for:
- Loading ROOT files with uproot
- Preprocessing particle physics data
- Building feature pairs for ML models
- Splitting data for training/validation

Configuration:
- LoadConfig: Used during data loading (file paths, cuts, truth labels)
- DataConfig: Describes loaded data structure (passed to ML models)
"""

import numpy as np
import pandas as pd
import yaml
import copy

from typing import Optional, Dict, List, Tuple, Union

from ..configs import LoadConfig, DataConfig


class LabelBuilder:
    """Builds training labels from truth information."""

    def __init__(
        self,
        load_config: LoadConfig,
        data: Union[pd.DataFrame, tuple[np.ndarray, np.ndarray]],
    ):
        self.config = load_config
        self.data = data

    def build_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build labels from jet and lepton truth information.

        Returns:
            pair_truth: Binary pairing labels (n_events, max_jets, NUM_LEPTONS)
            reco_mask: Boolean mask for successfully reconstructed events
        """
        jet_truth = self._extract_jet_truth()
        lepton_truth = self._extract_lepton_truth()
        reco_mask = self._get_reconstruction_mask(jet_truth)

        # Filter truth by reconstruction mask
        jet_truth = jet_truth[reco_mask]
        lepton_truth = lepton_truth[reco_mask]

        pair_truth = self._build_pair_truth_tensor(jet_truth, lepton_truth)

        return pair_truth, reco_mask

    def _extract_jet_truth(self) -> np.ndarray:
        """Extract jet truth indices."""
        if isinstance(self.data, pd.DataFrame):
            cols = [f"{self.config.jet_truth_label}_{i}" for i in [0, 3]]
            return self.data[cols].to_numpy()
        else:
            return self.data[0][:, [0, 3]]

    def _extract_lepton_truth(self) -> np.ndarray:
        """Extract lepton truth indices."""
        if isinstance(self.data, pd.DataFrame):
            cols = [f"{self.config.lepton_truth_label}_{i}" for i in [0, 1]]
            return self.data[cols].to_numpy()
        else:
            return self.data[1]

    def _get_reconstruction_mask(self, jet_truth: np.ndarray) -> np.ndarray:
        """Get mask for events with valid reconstruction."""
        valid_indices = (jet_truth != -1).all(axis=1)
        in_range = (jet_truth < self.config.max_jets).all(axis=1)
        return valid_indices & in_range

    def _build_pair_truth_tensor(
        self, jet_truth: np.ndarray, lepton_truth: np.ndarray
    ) -> np.ndarray:
        """Build binary tensor indicating correct jet-lepton pairs."""
        n_events = len(jet_truth)
        pair_truth = np.zeros(
            (n_events, self.config.max_jets, self.config.NUM_LEPTONS), dtype=np.float32
        )

        for top_idx in range(self.config.NUM_LEPTONS):
            jet_idx_for_lep = jet_truth[:, top_idx]  # (n_events,)
            lep_idx_for_lep = lepton_truth[:, top_idx]  # (n_events,)
            valid = jet_idx_for_lep >= 0
            evt = np.where(valid)[0]
            pair_truth[evt, jet_idx_for_lep[evt], lep_idx_for_lep[evt]] = 1
        return pair_truth


# =============================================================================
# Main Preprocessor
# =============================================================================


class TrainingDataLoader:
    """
    Main class for loading and preprocessing particle physics data.

    Handles:
    - Loading from NPZ files
    - Building feature arrays
    - Building labels


    After loading, provides a DataConfig that describes the loaded data
    structure for downstream components (e.g., ML models).
    """

    def __init__(self, load_config: LoadConfig):
        """
        Initialize preprocessor with loading configuration.

        Args:
            load_config: Configuration for data loading
        """
        self.load_config = load_config
        self.data_config: Optional[DataConfig] = None

        self.data = None
        self.data_length = None
        self.feature_data = None
        self.cut_dict = {}
        self.data_normalisation_factors = {}

    # -------------------------------------------------------------------------
    # Data Loading Helpers
    # -------------------------------------------------------------------------

    def _load_feature_array(
        self,
        loaded: Dict,
        feature_keys: List[str],
        target_shape: Optional[Tuple[int, ...]] = None,
        max_objects: Optional[int] = None,
    ) -> np.ndarray:
        """Helper to load and transpose feature arrays from npz.

        Args:
            loaded: Loaded npz file data
            feature_keys: List of keys to extract from npz
            target_shape: Shape to reshape into (if needed)
            max_objects: Maximum number of objects to keep (for jets/leptons)

        Returns:
            Transposed and reshaped feature array
        """
        arrays = [loaded[key] for key in feature_keys]
        result = (
            np.array(arrays).transpose(1, 2, 0)
            if len(arrays[0].shape) > 1
            else np.array(arrays).transpose(1, 0)
        )

        if max_objects is not None and len(result.shape) > 2:
            result = result[:, :max_objects, :]

        if target_shape is not None:
            result = result.reshape(target_shape)

        return result

    def _create_xy_dict(self, feature_data: Dict) -> Tuple[Dict, Dict]:
        """Helper to create X and y dictionaries from feature data.

        Args:
            feature_data: Dictionary of feature arrays

        Returns:
            X and y dictionaries
        """
        X = {k: v for k, v in feature_data.items() if v is not None}
        y = {
            "assignment": X["assignment"],
            "regression": X.get("regression", None),
        }
        return copy.deepcopy(X), copy.deepcopy(y)

    def _apply_mask_to_features(self, mask: np.ndarray) -> None:
        """Apply a boolean mask to all feature data.

        Args:
            mask: Boolean array to filter events
        """
        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][mask]

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    def load_from_npz(
        self,
        npz_path: Union[str, List[str]],
        num_events: Optional[int] = None,
        event_numbers: Optional[str] = None,
    ) -> DataConfig:
        """Load preprocessed data from NPZ file or list of files

        Args:
            npz_path: Path to NPZ file
            num_events: Maximum number of events to load
            event_numbers: Filter by 'even', 'odd','all', or None
        """
        if isinstance(npz_path, str):
            loaded = np.load(npz_path)
            mask = self._get_event_filter_mask(loaded, event_numbers)
            loaded_data = self._filter_loaded_data(loaded, mask, num_events)
        else:
            # If multiple paths are provided, load and concatenate them
            loaded_list = [np.load(path) for path in npz_path]
            mask_list = [
                self._get_event_filter_mask(loaded, event_numbers)
                for loaded in loaded_list
            ]
            loaded_data = {}
            for key in loaded_list[0].files:
                concatenated = np.concatenate(
                    [
                        self._filter_loaded_data(loaded, mask, num_events)[key]
                        for loaded, mask in zip(loaded_list, mask_list)
                    ],
                    axis=0,
                )
                loaded_data[key] = concatenated

        # Initialize feature data storage
        self.feature_data = {}

        # Load source features
        self._load_core_features(loaded_data)

        # Load optional features
        self._load_optional_features(loaded_data)

        # Load truth features
        self._load_truth_features(loaded_data)

        # Build labels and apply reconstruction mask
        self._build_and_apply_labels(loaded_data)

        # Remove NaN events from NuFlows if present
        self._filter_nuflows_nans()

        self.data_length = len(self.feature_data["assignment"])

        # Create DataConfig for downstream use
        self.data_config = self.load_config.to_data_config()
        return self.data_config

    def _get_event_filter_mask(
        self, loaded: Dict, event_numbers: Optional[str]
    ) -> np.ndarray:
        """Get mask for filtering events by event number parity."""
        if event_numbers is not None and self.load_config.mc_event_number is not None:
            event_number_array = loaded[self.load_config.mc_event_number]
            if event_numbers == "even":
                return (event_number_array % 2) == 0
            elif event_numbers == "odd":
                return (event_number_array % 2) == 1
            elif event_numbers == "all":
                return np.ones(len(event_number_array), dtype=bool)
            else:
                raise ValueError("event_numbers must be 'even', 'odd', or None")
        return np.ones(len(loaded[list(loaded.files)[0]]), dtype=bool)

    def _filter_loaded_data(
        self, loaded: Dict, mask: np.ndarray, num_events: Optional[int]
    ) -> Dict:
        """Apply mask and num_events limit to loaded data."""
        if num_events is not None:
            result = {}
            for key in loaded.files:
                result[key] = loaded[key][mask][:num_events]
            return result
        return loaded

    def _load_core_features(self, loaded: Dict) -> None:
        """Load source jet and lepton features."""
        if self.load_config.jet_inputs:
            self.feature_data["jet_inputs"] = self._load_feature_array(
                loaded,
                self.load_config.jet_inputs,
                max_objects=self.load_config.max_jets,
            )

        if self.load_config.lepton_inputs:
            self.feature_data["lep_inputs"] = self._load_feature_array(
                loaded, self.load_config.lepton_inputs
            )

    def _load_optional_features(self, loaded: Dict) -> None:
        """Load optional features (MET, global, non-training, weights)."""
        if self.load_config.global_event_inputs:
            global_event_data = self._load_feature_array(
                loaded, self.load_config.global_event_inputs
            )
            self.feature_data["global_event_inputs"] = global_event_data[
                :, np.newaxis, :
            ]

        if self.load_config.met_inputs:
            met_data = self._load_feature_array(loaded, self.load_config.met_inputs)
            self.feature_data["met_inputs"] = met_data[:, np.newaxis, :]

        if self.load_config.non_training_features:
            self.feature_data["non_training"] = self._load_feature_array(
                loaded, self.load_config.non_training_features
            )

        if self.load_config.event_weight:
            self.feature_data["event_weight"] = loaded[self.load_config.event_weight]

        if self.load_config.mc_event_number:
            self.feature_data["event_number"] = loaded[self.load_config.mc_event_number]

    def _load_truth_features(self, loaded: Dict) -> None:
        """Load truth features for neutrinos, tops, and leptons."""
        # Neutrino momentum truth
        if (
            self.load_config.neutrino_momentum_features
            and self.load_config.antineutrino_momentum_features
        ):
            combined_keys = (
                self.load_config.neutrino_momentum_features
                + self.load_config.antineutrino_momentum_features
            )
            data_length = len(loaded[combined_keys[0]])
            target_shape = (data_length, self.load_config.NUM_LEPTONS, -1)
            self.feature_data["regression"] = self._load_feature_array(
                loaded, combined_keys, target_shape=target_shape
            )

        # NuFlows neutrino truth
        if (
            self.load_config.nu_flows_neutrino_momentum_features
            and self.load_config.nu_flows_antineutrino_momentum_features
        ):
            combined_keys = (
                self.load_config.nu_flows_neutrino_momentum_features
                + self.load_config.nu_flows_antineutrino_momentum_features
            )
            data_length = len(loaded[combined_keys[0]])
            target_shape = (data_length, self.load_config.NUM_LEPTONS, -1)
            self.feature_data["nu_flows_neutrino_regression"] = (
                self._load_feature_array(
                    loaded, combined_keys, target_shape=target_shape
                )
            )

        # Top truth
        if self.load_config.top_truth_features and self.load_config.tbar_truth_features:
            combined_keys = (
                self.load_config.top_truth_features
                + self.load_config.tbar_truth_features
            )
            data_length = len(loaded[combined_keys[0]])
            target_shape = (data_length, self.load_config.NUM_LEPTONS, -1)
            self.feature_data["top_truth"] = self._load_feature_array(
                loaded, combined_keys, target_shape=target_shape
            )

        # Lepton truth
        if (
            self.load_config.top_lepton_truth_features
            and self.load_config.tbar_lepton_truth_features
        ):
            combined_keys = (
                self.load_config.top_lepton_truth_features
                + self.load_config.tbar_lepton_truth_features
            )
            data_length = len(loaded[combined_keys[0]])
            target_shape = (data_length, self.load_config.NUM_LEPTONS, -1)
            self.feature_data["lepton_truth"] = self._load_feature_array(
                loaded, combined_keys, target_shape=target_shape
            )

    def _build_and_apply_labels(self, loaded: Dict) -> None:
        """Build labels and apply reconstruction mask."""
        label_builder = LabelBuilder(
            self.load_config,
            (
                loaded[self.load_config.jet_truth_label],
                loaded[self.load_config.lepton_truth_label],
            ),
        )
        assignment, reco_mask = label_builder.build_labels()

        # Apply reconstruction mask to all features
        self._apply_mask_to_features(reco_mask)
        self.feature_data["assignment"] = assignment

    def _filter_nuflows_nans(self) -> None:
        """Remove events with NaNs in NuFlows targets if present."""
        if "nu_flows_neutrino_regression" in self.feature_data:
            nu_flows_nan_mask = ~np.isnan(
                self.feature_data["nu_flows_neutrino_regression"]
            ).any(axis=(1, 2))
            self._apply_mask_to_features(nu_flows_nan_mask)

    def get_data_config(self) -> DataConfig:
        """
        Get the DataConfig describing loaded data structure.

        This should be passed to downstream components like ML models.

        Returns:
            DataConfig instance

        Raises:
            ValueError: If data has not been loaded yet
        """
        if self.data_config is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data_config

    def _filter_negative_weights(self, cut_neg_weights: bool) -> None:
        """Remove events with negative weights if requested."""
        if self.load_config.event_weight and cut_neg_weights:
            n_before = len(self.data)
            self.data = self.data[self.data[self.load_config.event_weight] >= 0]
            n_after = len(self.data)
            if n_before != n_after:
                print(f"Removed {n_before - n_after} events with negative weights")

    # -------------------------------------------------------------------------
    # Custom Features
    # -------------------------------------------------------------------------

    def add_custom_feature(self, function: callable, name: str) -> None:
        """
        Add a custom feature computed from existing features.

        Args:
            function: Function that takes feature_data dict and returns new feature
            name: Name for the new feature
        """
        if self.feature_data is None:
            raise ValueError("Feature data not loaded. Call load_data() first.")
        if self.data_config is None:
            raise ValueError("DataConfig not available. Call load_data() first.")

        new_data = function(self.feature_data, self.data_config)

        if len(new_data) != self.data_length:
            raise ValueError(
                f"Custom feature must have {self.data_length} events, "
                f"got {len(new_data)}"
            )

        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        # Add to feature data
        if "custom" not in self.feature_data:
            self.feature_data["custom"] = new_data
            feature_idx = 0
        else:
            feature_idx = self.feature_data["custom"].shape[1]
            self.feature_data["custom"] = np.concatenate(
                (self.feature_data["custom"], new_data), axis=1
            )
        self.data_config.add_custom_feature(name, feature_idx)

    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def get_feature_data(self, data_type: str, feature_name: str) -> np.ndarray:
        """
        Get specific feature data.

        Args:
            data_type: Type of feature ('jet', 'lepton', 'met', 'non_training', etc.)
            feature_name: Name of specific feature

        Returns:
            Feature array (shape depends on data_type)
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.data_config is None:
            raise ValueError("DataConfig not available. Call load_data() first.")

        # Get feature index from DataConfig
        feature_idx = self.data_config.get_feature_index(data_type, feature_name)

        if data_type in [
            "jet_inputs",
            "lep_inputs",
            "met_inputs",
            "global_event_inputs",
        ]:
            return self.feature_data[data_type][:, :, feature_idx].copy()
        elif data_type in ["non_training", "custom"]:
            return self.feature_data[data_type][:, feature_idx].copy()
        elif data_type == "event_weight":
            return self.get_event_weight().copy()
        elif data_type == "event_number":
            return self.get_event_number().copy()
        elif data_type in self.feature_data:
            return self.feature_data[data_type][..., feature_idx].copy()
        else:
            raise ValueError(f"Data type '{data_type}' not found in feature data")

    def get_all_feature_data(self, feature_type: str) -> np.ndarray:
        """Get all features of a given type."""
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if feature_type not in self.feature_data:
            raise ValueError(f"Feature type '{feature_type}' not found")

        return self.feature_data[feature_type].copy()

    def get_event_weight(self, cut_neg_weights: bool = False) -> np.ndarray:
        """
        Get normalized event weights.

        Args:
            cut_neg_weights: Whether to exclude negative weights

        Returns:
            Normalized event weights
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.load_config.event_weight is None:
            raise ValueError("Event weight not configured")

        weights = self.feature_data["event_weight"]

        if cut_neg_weights:
            weights = weights[weights >= 0]

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        return weights

    def get_event_number(self) -> np.ndarray:
        """
        Get event numbers.

        Returns:
            Event numbers
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.load_config.mc_event_number is None:
            raise ValueError("Event number not configured")

        event_numbers = self.feature_data["event_number"]

        return event_numbers

    # -------------------------------------------------------------------------
    # Data Splitting
    # -------------------------------------------------------------------------

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, y_train, X_test, y_test
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        from sklearn.model_selection import train_test_split

        X_train, X_test = {}, {}

        for key, data in self.feature_data.items():
            if data is not None:
                X_train[key], X_test[key] = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )

        _, y_train = self._create_xy_dict(X_train)
        _, y_test = self._create_xy_dict(X_test)

        return X_train, y_train, X_test, y_test

    def get_num_events(self) -> int:
        """Get the number of events in the loaded dataset."""
        if self.data_length is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data_length

    def get_data(self):
        return self._create_xy_dict(self.feature_data)

    def get_k_fold(
        self, n_folds: int = 5, k_fold_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get a specific fold for k-fold cross-validation.

        Args:
            n_folds: Total number of folds
            k_fold_idx: Index of the fold to retrieve (0-based)
        Returns:
            X_fold, y_fold for the specified fold
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        # Create folds
        folds = []
        start = 0
        end = self.data_length - (self.data_length % n_folds)  # Exclude leftover events
        fold_size = end // n_folds

        test_start = start + k_fold_idx * fold_size
        test_end = test_start + fold_size if k_fold_idx < n_folds - 1 else end

        # Training data: everything except test fold
        train_indices = np.concatenate(
            [np.arange(start, test_start), np.arange(test_end, end)]
        )
        test_indices = np.arange(test_start, test_end)

        X_train = {k: self.feature_data[k][train_indices] for k in self.feature_data}
        X_test = {k: self.feature_data[k][test_indices] for k in self.feature_data}

        return *self._create_xy_dict(X_train), *self._create_xy_dict(X_test)

    def create_k_folds(
        self, n_folds: int = 5, n_splits: int = 1, random_state: int = 42
    ) -> List[
        Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]
    ]:
        """
        Create k-fold cross-validation splits.

        Args:
            n_folds: Number of folds
            n_splits: Number of independent splits
            random_state: Random seed

        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        if random_state is not None:
            np.random.seed(random_state)

        # Shuffle data once
        indices = np.arange(self.data_length)
        np.random.shuffle(indices)

        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][indices]

        # Create folds
        split_size = self.data_length // n_splits
        folds = []

        for split_idx in range(n_splits):
            start = split_idx * split_size
            end = (split_idx + 1) * split_size

            split_len = end - start
            fold_size = split_len // n_folds

            for fold_idx in range(n_folds):
                test_start = start + fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < n_folds - 1 else end

                # Training data: everything except test fold
                train_indices = np.concatenate(
                    [np.arange(start, test_start), np.arange(test_end, end)]
                )
                test_indices = np.arange(test_start, test_end)

                X_train = {
                    k: self.feature_data[k][train_indices] for k in self.feature_data
                }
                X_test = {
                    k: self.feature_data[k][test_indices] for k in self.feature_data
                }

                _, y_train = self._create_xy_dict(X_train)
                _, y_test = self._create_xy_dict(X_test)

                folds.append((X_train, y_train, X_test, y_test))

        return folds

    def split_even_odd(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """
        Split data into even and odd event number sets.

        Returns:
            X_even, y_even, X_odd, y_odd
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        if "event_number" not in self.feature_data:
            raise ValueError("Event number feature not available for splitting.")

        event_numbers = self.feature_data["event_number"]
        even_mask = (event_numbers % 2) == 0
        odd_mask = ~even_mask

        X_even = {k: v[even_mask] for k, v in self.feature_data.items()}
        X_odd = {k: v[odd_mask] for k, v in self.feature_data.items()}

        _, y_even = self._create_xy_dict(X_even)
        _, y_odd = self._create_xy_dict(X_odd)

        return X_even, y_even, X_odd, y_odd


def load_yaml_config(file_path):
    """Load a YAML configuration file."""
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_load_config_from_yaml(file_path: str) -> LoadConfig:
    """
    Load LoadConfig from a YAML file.

    Args:
        file_path: Path to YAML configuration file
    Returns:
        LoadConfig instance
    """

    config_dict = load_yaml_config(file_path)
    if "LoadConfig" not in config_dict:
        raise ValueError("YAML file must contain a 'LoadConfig' section")
    return LoadConfig(**config_dict["LoadConfig"])


def combine_train_datasets(
    X: List[dict[str : np.ndarray]],
    y: List[dict[str : np.ndarray]],
    weights: Optional[List[float]] = None,
) -> dict[str : np.ndarray]:
    """
    Combine multiple training datasets into one.

    Args:
        X: List of feature dictionaries
        y: List of label dictionaries
    Returns:
        Combined feature dictionary
        combined_X = {}
        combined_y = {}
    """
    combined_X = {}
    combined_y = {}

    for key in X[0].keys():
        combined_X[key] = np.concatenate([x[key] for x in X], axis=0)

    for key in y[0].keys():
        combined_y[key] = np.concatenate([label[key] for label in y], axis=0)

    if weights is not None:
        # Apply weights to event weights if present
        if "event_weight" in combined_X:
            event_weights = []
            for i, x in enumerate(X):
                n_events = x["event_weight"].shape[0]
                event_weights.append(
                    x["event_weight"] / np.mean(x["event_weight"]) * weights[i]
                )
            combined_X["event_weight"] = np.concatenate(event_weights, axis=0)

    permutation = np.random.permutation(combined_X[list(combined_X.keys())[0]].shape[0])
    for key in combined_X.keys():
        combined_X[key] = combined_X[key][permutation]
    for key in combined_y.keys():
        combined_y[key] = combined_y[key][permutation]

    return combined_X, combined_y
