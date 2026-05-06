import numpy as np
from ..configs import DataConfig
import keras


def compute_sample_weights(X_train: dict, data_config: DataConfig) -> np.ndarray:
    """
    Compute sample weights based on the number of jets in each event.
    Events with more jets are given higher weights to balance the training.

    Args:
        X_train (dict): Dictionary containing training features
        y_train (np.ndarray): Array of true labels with shape (num_samples, max_jets, 2).
    Returns:
        sample_weights (np.ndarray): Array of sample weights with shape (num_samples,).
    """
    padding_value = data_config.padding_value
    event_weights = np.ones(X_train[list(X_train.keys())[0]].shape[0])
    jet_inputs = None
    truth_ttbar_mass = None
    for key in X_train.keys():
        if "non_training" in key:
            if "truth_ttbar_mass" in data_config.feature_indices[key]:
                truth_ttbar_mass_index = data_config.feature_indices[key][
                    "truth_ttbar_mass"
                ]
                truth_ttbar_mass = X_train[key][truth_ttbar_mass_index].flatten()
        if "event_weight" in key:
            event_weights = X_train[key].flatten()
        if "jet_inputs" in key:
            jet_inputs = X_train[key]
            break
    if jet_inputs is None:
        raise ValueError("Jet data not found in X_train.")

    if truth_ttbar_mass is not None:
        hist, bins = np.histogram(truth_ttbar_mass, bins=20)
        bin_indices = np.digitize(truth_ttbar_mass, bins) - 1
        bin_weights = 1.0 / (hist + 1e-6)
        event_weights *= bin_weights[bin_indices]

    # Count valid jets per event (assuming padding value is -999)
    padding_value = -999
    valid_jets = np.sum(
        np.any(jet_inputs != padding_value, axis=-1), axis=-1
    )  # (num_samples,)

    num_jets = np.unique(valid_jets)

    for n in num_jets:
        mask = valid_jets == n
        # event_weights[mask] *= (1.0 / np.sum(mask))  # Weight inversely proportional to count

    # Normalize weights to have mean of 1
    event_weights /= np.mean(event_weights)
    return event_weights


def prepare_model_inputs(model: keras.Model, inputs: dict) -> dict:
    """
    Prepare model inputs by selecting the relevant features based on the model's input layers.

    Args:
        model (keras.Model): The Keras model for which to prepare inputs.
        X_train (dict): Dictionary containing training features.

    Returns:
        model_inputs (dict): Dictionary containing only the features required by the model.
    """
    model_inputs = {}
    for input_name in model.input:
        if not isinstance(input_name, str):
            input_name = input_name.name.split(":")[0]
        if input_name in inputs:
            model_inputs[input_name] = inputs[input_name]
        else:
            raise ValueError(f"Input '{input_name}' not found in X_train.")
    return model_inputs


def get_jet_mask(jet_inputs: np.ndarray, padding_value: float = -999) -> np.ndarray:
    """
    Get a boolean mask indicating which jets are valid (not padded).

    Args:
        jet_inputs (np.ndarray): Array of shape (num_samples, max_jets, num_features) containing jet features.
        padding_value (float): The value used for padding invalid jets.

    Returns:
        jet_mask (np.ndarray): Boolean array of shape (num_samples, max_jets) where True indicates a valid jet.
    """
    return np.any(jet_inputs != padding_value, axis=-1)
