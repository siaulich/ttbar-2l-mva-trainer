"""
Configuration classes for data loading and ML pipeline.

This module separates loading configuration (LoadConfig) from data structure
description (DataConfig). LoadConfig is used during data loading, while
DataConfig describes the loaded data structure and is passed to downstream
components like ML models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import keras

@dataclass
class RecontructorConfig:
    type: str = "KerasFFRecoBase"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BinningVariableConfig:
    feature_type: str
    feature_name: str
    fancy_feature_label: str
    bins: Optional[int] = None
    xlims: Optional[Tuple[float, float]] = None
    rescale_factor: Optional[float] = None
    center_bins: bool = False


@dataclass
class EvaluationConfig:
    reconstructors: List[RecontructorConfig] = field(default_factory=list)
    evaluation_event_numbers: str = "odd"
    binning_variables: List[BinningVariableConfig] = field(default_factory=list)
    binned_2d_binning_variables: List[List[BinningVariableConfig]] = field(default_factory=list)


@dataclass
class LoadConfig:
    """
    Configuration for loading data from ROOT files.

    Contains all options needed during the data loading phase, including
    feature names, truth labels, cuts, and preprocessing options.

    This config is used by the DataPreprocessor during loading.
    After loading, a DataConfig is created to describe the loaded data structure.
    """

    # Data source
    data_dir: str
    data_path: Dict[str, str]

    # Feature specifications
    jet_features: List[str]
    lepton_features: List[str]
    jet_truth_label: str
    lepton_truth_label: str

    # MC truth
    top_truth_features: Optional[List[str]] = None
    tbar_truth_features: Optional[List[str]] = None
    top_lepton_truth_features: Optional[List[str]] = None
    tbar_lepton_truth_features: Optional[List[str]] = None

    # Maximum objects per event
    NUM_LEPTONS: int = 2
    max_jets: int = 4

    # Optional features
    met_features: Optional[List[str]] = None
    global_event_inputs: Optional[List[str]] = None
    non_training_features: Optional[List[str]] = None

    # Regression target features
    neutrino_momentum_features: Optional[List[str]] = None
    antineutrino_momentum_features: Optional[List[str]] = None
    nu_flows_neutrino_momentum_features: Optional[List[str]] = None
    nu_flows_antineutrino_momentum_features: Optional[List[str]] = None
    event_weight: Optional[str] = None
    mc_event_number: Optional[str] = None

    # Loading options
    padding_value: float = -999.0

    def to_data_config(self) -> "DataConfig":
        """
        Create a DataConfig from this LoadConfig.

        Returns:
            DataConfig describing the structure of loaded data
        """
        return DataConfig(
            jet_features=self.jet_features,
            lepton_features=self.lepton_features,
            met_features=self.met_features,
            non_training_features=self.non_training_features,
            max_jets=self.max_jets,
            NUM_LEPTONS=self.NUM_LEPTONS,
            padding_value=self.padding_value,
            has_neutrino_truth=self.neutrino_momentum_features is not None,
            neutrino_momentum_features=self.neutrino_momentum_features,
            antineutrino_momentum_features=self.antineutrino_momentum_features,
            has_nu_flows_neutrino_truth=self.nu_flows_neutrino_momentum_features
            is not None,
            nu_flows_neutrino_momentum_features=self.nu_flows_neutrino_momentum_features,
            nu_flows_antineutrino_momentum_features=self.nu_flows_antineutrino_momentum_features,
            top_truth_features=self.top_truth_features,
            tbar_truth_features=self.tbar_truth_features,
            top_lepton_truth_features=self.top_lepton_truth_features,
            tbar_lepton_truth_features=self.tbar_lepton_truth_features,
            global_event_inputs=self.global_event_inputs,
            has_global_event_inputs=self.global_event_inputs is not None,
            has_top_truth=self.top_truth_features is not None,
            has_lepton_truth=self.top_lepton_truth_features is not None,
            has_event_weight=self.event_weight is not None,
            has_event_number=self.mc_event_number is not None,
        )

def load_yaml_config(yaml_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file
    Returns:
        Dictionary containing the loaded configuration
    """
    import yaml

    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def get_load_config_from_yaml(yaml_path: str) -> LoadConfig:
    """
    Load a LoadConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file
    Returns:
        LoadConfig instance
    """
    import yaml

    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    config_dict = yaml_dict.get("LoadConfig", {})

    return LoadConfig(**config_dict)


@dataclass
class DataConfig:
    """
    Configuration describing the structure of loaded data.

    This class describes how the data looks after loading, including feature
    names, shapes, and indices. It should be passed to downstream components
    like ML models, training loops, and evaluation scripts.

    This config does NOT contain loading-specific options like file paths,
    truth labels, or cuts - those are in LoadConfig.

    Attributes:
        jet_features: Names of jet features
        lepton_features: Names of lepton features
        met_features: Names of MET features (optional)
        non_training_features: Names of features not used in training (optional)
        max_jets: Maximum number of jets per event
        NUM_LEPTONS: Maximum number of leptons per event
        padding_value: Value used for padding
        has_neutrino_truth: Whether regression targets are present
        regression_target_full_reco_names: Names of regression targets (optional)
        has_event_weight: Whether event weights are present
        feature_indices: Dictionary mapping feature names to their indices
        data_shapes: Dictionary describing array shapes for each feature type
        custom_features: Custom features added after loading
    """

    # Feature names
    jet_features: List[str]
    lepton_features: List[str]
    met_features: Optional[List[str]] = None
    global_event_inputs: Optional[List[str]] = None
    has_global_event_inputs: bool = False

    # Non-training features
    non_training_features: Optional[List[str]] = None
    custom_features: Dict[str, int] = field(default_factory=dict)

    # Data structure
    max_jets: int = 4
    NUM_LEPTONS: int = 2
    padding_value: float = -999.0

    # Optional components
    has_neutrino_truth: bool = False
    neutrino_momentum_features: Optional[List[str]] = None
    antineutrino_momentum_features: Optional[List[str]] = None

    # Nu-flows regression targets
    has_nu_flows_neutrino_truth: bool = False
    nu_flows_neutrino_momentum_features: Optional[List[str]] = None
    nu_flows_antineutrino_momentum_features: Optional[List[str]] = None

    # MC truth
    top_truth_features: Optional[List[str]] = None
    tbar_truth_features: Optional[List[str]] = None
    top_lepton_truth_features: Optional[List[str]] = None
    tbar_lepton_truth_features: Optional[List[str]] = None

    has_top_truth: bool = False
    has_lepton_truth: bool = False

    # Event weights and numbers
    has_event_weight: bool = False
    has_event_number: bool = False

    # Computed properties (populated after loading)
    feature_indices: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    data_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict, init=False)
    custom_features: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize computed properties."""
        self._build_feature_indices()
        self._build_data_shapes()

    def _add_feature_indices(self, key: str, features: Optional[List[str]]) -> None:
        """Helper to add feature indices for a given feature type."""
        if features:
            self.feature_indices[key] = {var: idx for idx, var in enumerate(features)}

    def _build_feature_indices(self) -> None:
        """Build index mapping for all feature types."""
        self.feature_indices = {}

        # Core features (always present)
        self._add_feature_indices("lep_inputs", self.lepton_features)
        self._add_feature_indices("jet_inputs", self.jet_features)
        self._add_feature_indices("met_inputs", self.met_features)

        # Optional features
        self._add_feature_indices("non_training", self.non_training_features)

        # Single-value features
        if self.has_event_weight:
            self.feature_indices["event_weight"] = {"weight": 0}
        if self.has_event_number:
            self.feature_indices["event_number"] = {"event_number": 0}

        # Truth features (conditional)
        if self.has_neutrino_truth:
            self._add_feature_indices("regression", self.neutrino_momentum_features)
        if self.has_nu_flows_neutrino_truth:
            self._add_feature_indices(
                "nu_flows_neutrino_truth", self.nu_flows_neutrino_momentum_features
            )
        if self.has_global_event_inputs:
            self._add_feature_indices("global_event_inputs", self.global_event_inputs)
        if self.has_top_truth:
            self._add_feature_indices("top_truth", self.top_truth_features)
        if self.has_lepton_truth:
            self._add_feature_indices("lepton_truth", self.top_lepton_truth_features)

    def _build_data_shapes(self) -> None:
        """Build expected data shapes for each feature type."""
        self.data_shapes = {}

        # Define shape configurations: (key, condition, feature_list, shape_dims)
        shape_configs = [
            # Core object features (n_events, n_objects, n_features)
            ("lep_inputs", True, self.lepton_features, (None, self.NUM_LEPTONS)),
            ("jet_inputs", True, self.jet_features, (None, self.max_jets)),
            ("met_inputs", self.met_features, self.met_features, (None, 1)),
            # Event-level features (n_events, n_features)
            (
                "non_training",
                self.non_training_features,
                self.non_training_features,
                (None,),
            ),
            (
                "global_event_inputs",
                self.has_global_event_inputs,
                self.global_event_inputs,
                (None,),
            ),
            # Truth features (n_events, NUM_LEPTONS, n_features)
            (
                "regression",
                self.has_neutrino_truth,
                self.neutrino_momentum_features,
                (None, self.NUM_LEPTONS),
            ),
            (
                "nu_flows_neutrino_truth",
                self.has_nu_flows_neutrino_truth,
                self.nu_flows_neutrino_momentum_features,
                (None, self.NUM_LEPTONS),
            ),
            (
                "top_truth",
                self.has_top_truth,
                self.top_truth_features,
                (None, self.NUM_LEPTONS),
            ),
            (
                "lepton_truth",
                self.has_lepton_truth,
                self.top_lepton_truth_features,
                (None, self.NUM_LEPTONS),
            ),
        ]

        # Build shapes based on configuration
        for key, condition, features, base_shape in shape_configs:
            if condition and features:
                self.data_shapes[key] = base_shape + (len(features),)

        # Single-value features (n_events,)
        if self.has_event_weight:
            self.data_shapes["event_weight"] = (None,)
        if self.has_event_number:
            self.data_shapes["event_number"] = (None,)

        # Labels shape
        self.data_shapes["labels"] = (None, self.max_jets, self.NUM_LEPTONS)

    # =========================================================================
    # Accessors for downstream evaluation components
    # =========================================================================
    def get_feature_index(self, feature_type: str, feature_name: str) -> int:
        """
        Get the index of a specific feature.

        Args:
            feature_type: Type of feature ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature

        Returns:
            Index of the feature

        Raises:
            KeyError: If feature type or name not found
        """
        return self.feature_indices[feature_type][feature_name]

    def add_custom_feature(self, name: str, index: int) -> None:
        """
        Register a custom feature.

        Args:
            name: Name of the custom feature
            index: Index in the custom feature array
        """
        self.custom_features[name] = index
        if "custom" not in self.feature_indices:
            self.feature_indices["custom"] = {}
        self.feature_indices["custom"][name] = index
