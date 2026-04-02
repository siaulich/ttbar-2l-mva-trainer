import keras as keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class SplitTransformerOutput(keras.layers.Layer):
    def __init__(
        self, name="SplitTransformerOutput", max_jets=6, max_leptons=2, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.max_jets = max_jets
        self.max_leptons = max_leptons

    def call(self, inputs):
        """
        Splits the transformer output into jet, lepton, and MET components.
        Args:
            inputs (tf.Tensor): The output tensor from the transformer of shape (batch_size, seq_len, hidden_dim).
        Returns:
            tuple: A tuple containing:
                - jet_outputs (tf.Tensor): The jet component of shape (batch_size, max_jets, hidden_dim).
                - lepton_outputs (tf.Tensor): The lepton component of shape (batch_size, max_leptons, hidden_dim).
                - met_outputs (tf.Tensor): The MET component of shape (batch_size, 1, hidden_dim).
        """
        jet_outputs = inputs[:, : self.max_jets, :]
        lepton_outputs = inputs[:, self.max_jets : self.max_jets + self.max_leptons, :]
        met_outputs = inputs[:, -1:, :]
        return jet_outputs, lepton_outputs, met_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_jets": self.max_jets,
                "max_leptons": self.max_leptons,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class ConcatLeptonCharge(keras.layers.Layer):
    def __init__(self, name="ConcatLeptonCharge", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, input):
        """
        Concatenates lepton charge information to lepton features.
        Args:
            input: tensor of shape (batch_size, max_leptons, feature_dim) where the last dimension is the charge.

        Returns:
            tf.Tensor: Tensor of shape (batch_size, max_leptons, feature_dim + 1) with charge concatenated.
        """
        lepton_inputs = input[:, :, :-1]
        # Create charge indicators: 1 for first lepton, -1 for second lepton
        charge_indicators = tf.constant([[1.0], [-1.0]], dtype=lepton_inputs.dtype)
        charge_indicators = tf.reshape(charge_indicators, [1, 2, 1])
        charge_indicators = tf.tile(charge_indicators, [tf.shape(input)[0], 1, 1])
        concatenated = tf.concat([lepton_inputs, charge_indicators], axis=-1)
        return concatenated


@keras.utils.register_keras_serializable()
class ExpandJetMask(keras.layers.Layer):
    def __init__(self, name="ExpandJetMask", extra_sequence_length=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.extra_sequence_length = extra_sequence_length

    def call(self, jet_mask):
        """
        Expands the jet mask to match the sequence length of the transformer output.
        Args:
            jet_mask (tf.Tensor): The jet mask tensor of shape (batch_size, max_jets).
        Returns:
            tf.Tensor: The expanded jet mask tensor of shape (batch_size, max_jets + extra_sequence_length).
        """
        batch_size = tf.shape(jet_mask)[0]
        extra_mask = tf.ones(
            (batch_size, self.extra_sequence_length), dtype=jet_mask.dtype
        )
        expanded_mask = tf.concat([jet_mask, extra_mask], axis=1)
        return expanded_mask

    def get_config(self):
        config = super().get_config()
        config.update({"extra_sequence_length": self.extra_sequence_length})
        return config


import numpy as np


@keras.utils.register_keras_serializable()
class UnbinRegressionOutput(keras.layers.Layer):
    def __init__(self, scale, name="UnbinRegressionOutput", **kwargs):
        """
        Args:
            scale: float or numpy array defining the full output range.
                   The output range will be approximately [-0.5*scale, 0.5*scale].
        """
        super().__init__(name=name, **kwargs)
        self.scale = np.array(scale)

    def build(self, input_shape):
        # Number of bins inferred from last axis
        self.n_bins = int(input_shape[-1])

        # Precompute bin centers in [0, 1]
        bin_indices = tf.range(self.n_bins, dtype=self.compute_dtype)
        self.bin_centers = (bin_indices + 0.5) / self.n_bins  # bin centers in [0,1]

        super().build(input_shape)

    def call(self, inputs):
        """
        Converts binned logits/probabilities to continuous value using argmax.
        Args:
            inputs: Tensor of shape (..., n_bins)
        Returns:
            Tensor of shape (...)
        """
        dtype = self.compute_dtype
        scale = tf.cast(self.scale, dtype)

        # Hard bin selection (inference only)
        bin_idx = tf.argmax(inputs, axis=-1, output_type=tf.int32)
        bin_idx = tf.cast(bin_idx, dtype)

        # Map to bin center in [0,1]
        normalized = (bin_idx + 0.5) / tf.cast(self.n_bins, dtype)

        # Shift to [-0.5, 0.5] and scale
        return (normalized - 0.5) * scale

    def bin_data(self, regression_data):
        """
        Converts continuous regression targets into one-hot bins.

        Args:
            regression_data: Tensor of shape (...)

        Returns:
            One-hot tensor of shape (..., n_bins)
        """
        dtype = self.compute_dtype
        scale = tf.cast(self.scale, dtype)

        # Normalize to [0,1]
        normalized = regression_data / scale + 0.5

        # Convert to bin index
        bin_idx = tf.floor(normalized * self.n_bins)
        bin_idx = tf.clip_by_value(bin_idx, 0, self.n_bins - 1)
        bin_idx = tf.cast(bin_idx, tf.int32)
        return bin_idx

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale": self.scale.tolist(),  # JSON-safe
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["scale"] = np.array(config["scale"])
        return cls(**config)
