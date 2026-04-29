import keras as keras
import tensorflow as tf
import numpy as np


from ..reconstruction import KerasBinnedRegressor
from ..components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    PoolingAttentionBlock,
)


class FeatureConcatBinnedReconstructor(KerasBinnedRegressor):
    def __init__(
        self, config, name="Transformer", use_nu_flows=False, perform_regression=True
    ):
        super().__init__(
            config,
            name=name,
            perform_regression=False if use_nu_flows else perform_regression,
            use_nu_flows=use_nu_flows,
        )

    def build_model(
        self,
        hidden_dim,
        num_layers,
        dropout_rate,
        regression_bins=10,
        num_heads=8,
        compute_HLF=True,
        use_global_event_inputs=False,
        log_variables=False,
    ):
        """
        Builds the Assignment Transformer model.
        Args:
            hidden_dim (int): The dimensionality of the hidden layers.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            dropout_rate (float): The dropout rate to be applied in the model.
        Returns:
            keras.Model: The constructed Keras model.
        """
        # Input layers
        normed_inputs, masks = self._prepare_inputs(
            compute_HLF=compute_HLF,
            log_variables=log_variables,
            use_global_event_inputs=use_global_event_inputs,
        )
        normed_jet_inputs = normed_inputs["jet_inputs"]
        normed_lep_inputs = normed_inputs["lepton_inputs"]
        normed_met_inputs = normed_inputs["met_inputs"]
        jet_mask = masks["jet_mask"]

        if compute_HLF:
            normed_HLF_inputs = normed_inputs["hlf_inputs"]
            flat_normed_HLF_inputs = keras.layers.Reshape((self.max_jets, -1))(
                normed_HLF_inputs
            )
            normed_jet_inputs = keras.layers.Concatenate(axis=-1)(
                [normed_jet_inputs, flat_normed_HLF_inputs]
            )

        if "global_event_inputs" in normed_inputs:
            normed_global_event_inputs = normed_inputs["global_event_inputs"]
            flatted_global_event_inputs = keras.layers.Flatten()(
                normed_global_event_inputs
            )
            # Add global event features to jets
            global_event_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
                flatted_global_event_inputs
            )
            normed_jet_inputs = keras.layers.Concatenate(axis=-1)(
                [normed_jet_inputs, global_event_repeated_jets]
            )

        flatted_met_inputs = keras.layers.Flatten()(normed_met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(normed_lep_inputs)

        # Concat met and lepton features to each jet
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_inputs = keras.layers.Concatenate(axis=-1)(
            [normed_jet_inputs, met_repeated_jets, lepton_repeated_jets]
        )

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_inputs)

        # Transformer layers
        jets_transformed = jet_embedding
        for i in range(num_layers):
            jets_transformed = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
                pre_ln=True,
            )(jets_transformed, mask=jet_mask)

        # Output layers
        jet_output_embedding = MLP(
            self.NUM_LEPTONS,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        attention_pooling = PoolingAttentionBlock(
            num_heads=num_heads,
            key_dim=hidden_dim,
            num_seeds=1,
            dropout_rate=dropout_rate,
            name="attention_pooling",
        )(jets_transformed, mask=jet_mask)
        average_pooled = keras.layers.GlobalAveragePooling1D(name="average_pooling")(
            jets_transformed, mask=jet_mask
        )
        attention_pooling = keras.layers.Flatten()(attention_pooling)
        regression_outputs = keras.layers.Concatenate(name="regression_concat")(
            [attention_pooling, average_pooled]
        )

        regression_outputs = MLP(
            output_dim=regression_bins * self.NUM_LEPTONS * 3,
            name="regression_head_mlp",
            num_layers=4,
        )(regression_outputs)

        regression_outputs = keras.layers.Reshape(
            (self.NUM_LEPTONS, 3, regression_bins), name="regression_reshape"
        )(regression_outputs)

        jet_assignment_probs = TemporalSoftmax(name="assignment", axis=1)(
            jet_output_embedding, mask=jet_mask
        )
        binned_regression = keras.layers.Softmax(name="binned_regression", axis=-1)(
            regression_outputs
        )

        # Confidence score output (optional)
        self._build_model_base(
            jet_assignment_probs=jet_assignment_probs,
            regression_output=binned_regression,
            name="FeatureConcatBinnedTransformerModel",
        )
