import keras as keras
import tensorflow as tf
import numpy as np


from ..reconstruction import KerasFFRecoBase
from ..configs import DataConfig
from ..components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    PoolingAttentionBlock,
    ConcatLeptonCharge,
    ExpandJetMask,
    SplitTransformerOutput,
    JetLeptonAssignment,
    StopGradientLayer,
    EmbeddingMLP,
)


class FullRecoTransformer(KerasFFRecoBase):
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
        num_heads=8,
        use_global_event_inputs=False,
        compute_HLF=True,
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
        normed_inputs, masks = self._prepare_inputs(
            use_global_event_inputs=use_global_event_inputs,
            compute_HLF=compute_HLF,
            log_variables=log_variables,
        )
        normed_jet_inputs = normed_inputs["jet_inputs"]
        normed_lep_inputs = normed_inputs["lepton_inputs"]
        normed_met_inputs = normed_inputs["met_inputs"]
        jet_mask = masks["jet_mask"]

        # Embed jets
        jet_embeddings = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="jet_embedding_mlp",
            num_layers=4,
        )(normed_jet_inputs)

        # Embed leptons
        normed_lep_inputs = ConcatLeptonCharge()(normed_lep_inputs)
        lepton_embeddings = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="lepton_embedding_mlp",
            num_layers=4,
        )(normed_lep_inputs)

        # Embed MET
        met_embeddings = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="met_embedding_mlp",
            num_layers=4,
        )(normed_met_inputs)

        # Concatenate all embeddings
        combined_embeddings = keras.layers.Concatenate(axis=1)(
            [jet_embeddings, lepton_embeddings, met_embeddings]
        )

        x = combined_embeddings

        # Transformer layers
        self_attention_mask = ExpandJetMask(
            name="expand_jet_mask",
            extra_sequence_length=self.NUM_LEPTONS + 1,
        )(jet_mask)
        for i in range(num_layers):
            x = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"self_attention_block_{i}",
            )(x, self_attention_mask)

        # Split outputs
        jet_outputs, lepton_outputs, met_outputs = SplitTransformerOutput(
            name="split_transformer_output",
            max_jets=self.max_jets,
            max_leptons=self.NUM_LEPTONS,
        )(x)

        # Assignment Head
        jet_assignment_output = MLP(
            output_dim=hidden_dim,
            name="jet_assignment_mlp",
            num_layers=2,
        )(jet_outputs)
        lepton_assignment_output = MLP(
            output_dim=hidden_dim,
            name="lepton_assignment_mlp",
            num_layers=2,
        )(lepton_outputs)

        assignment_logits = JetLeptonAssignment(dim=hidden_dim, name="assignment")(
            jets=jet_assignment_output,
            leptons=lepton_assignment_output,
            jet_mask=jet_mask,
        )

        # Regression Head
        lepton_regression_outputs = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="lepton_regression_mlp",
            num_layers=2,
        )(lepton_outputs)

        regression_outputs = keras.layers.Concatenate(axis=1)(
            [lepton_regression_outputs, met_outputs]
        )
        regression_outputs = keras.layers.Flatten()(regression_outputs)
        regression_outputs = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="regression_hidden_mlp",
            num_layers=4,
        )(regression_outputs)
        regression_outputs = MLP(
            output_dim=3 * self.NUM_LEPTONS,
            name="regression_head_mlp",
            num_layers=4,
        )(regression_outputs)

        regression_outputs = keras.layers.Reshape(
            (-1, 3), name="normalized_regression"
        )(regression_outputs)

        self._build_model_base(
            assignment_logits,
            regression_outputs,
        )


class FeatureConcatReconstructor(KerasFFRecoBase):
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
        attention_pooling = keras.layers.Flatten()(attention_pooling)
        regression_outputs = keras.layers.Concatenate(name="regression_concat")(
            [attention_pooling]
        )

        regression_outputs = MLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="regression_hidden_mlp",
            num_layers=3,
        )(regression_outputs)
        regression_outputs = MLP(
            output_dim=3 * self.NUM_LEPTONS,
            name="regression_head_mlp",
            num_layers=2,
        )(regression_outputs)

        regression_outputs = keras.layers.Reshape(
            (-1, 3), name="normalized_regression"
        )(regression_outputs)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_output_embedding, mask=jet_mask
        )

        # Confidence score output (optional)
        self._build_model_base(
            jet_assignment_probs=jet_assignment_probs,
            regression_output=regression_outputs,
            name="FeatureConcatTransformerModel",
        )


class CompactReconstructor(KerasFFRecoBase):
    def __init__(self, config: DataConfig, name="CrossAttentionModel"):
        super().__init__(config, name=name)
        self.perform_regression = True

    def build_model(
        self,
        hidden_dim,
        num_layers,
        num_heads=8,
        dropout_rate=0.1,
        log_variables=False,
        compute_HLF=False,
        use_global_event_inputs=True,
        stop_gradient_assignment_probs=True,
    ):
        """
        Builds a more compact version of the Assignment Transformer model with fewer parameters.
        """
        # Input layers
        normed_inputs, masks = self._prepare_inputs(
            log_variables=log_variables,
            compute_HLF=compute_HLF,
            use_global_event_inputs=use_global_event_inputs,
        )

        normed_jet_inputs = normed_inputs["jet_inputs"]
        normed_lep_inputs = normed_inputs["lepton_inputs"]
        normed_met_inputs = normed_inputs["met_inputs"]
        normed_global_event_inputs = normed_inputs["global_event_inputs"]
        jet_mask = masks["jet_mask"]

        normed_global_inputs = keras.layers.Concatenate(axis=-1)(
            [normed_met_inputs, normed_global_event_inputs]
        )

        # Embed jets, leptons, and global event features
        jet_embedding = EmbeddingMLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="jet_embedding",
        )(normed_jet_inputs)

        lep_embedding = EmbeddingMLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="lep_embedding",
        )(normed_lep_inputs)

        global_embedding = EmbeddingMLP(
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name="global_embedding",
        )(normed_global_inputs)

        # Concatenate objects to sequence
        sequence = keras.layers.Concatenate(axis=1)(
            [jet_embedding, lep_embedding, global_embedding]
        )
        sequence_mask = ExpandJetMask(
            name="expand_jet_mask",
            extra_sequence_length=self.NUM_LEPTONS + 1,
        )(jet_mask)

        # Transformer layers
        x = sequence
        for i in range(num_layers):
            x = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"self_attention_block_{i}",
            )(x, sequence_mask)

        # Split outputs
        jet_outputs, lepton_outputs, global_outputs = SplitTransformerOutput(
            name="split_transformer_output",
            max_jets=self.max_jets,
            max_leptons=self.NUM_LEPTONS,
        )(x)

        # Assignment Head
        jet_assignment_output = MLP(
            output_dim=hidden_dim,
            name="jet_assignment_mlp",
            num_layers=2,
        )(jet_outputs)
        lepton_assignment_output = MLP(
            output_dim=hidden_dim,
            name="lepton_assignment_mlp",
            num_layers=2,
        )(lepton_outputs)
        assignment_probs = JetLeptonAssignment(dim=hidden_dim, name="assignment")(
            jets=jet_assignment_output,
            leptons=lepton_assignment_output,
            jet_mask=jet_mask,
        )
        # Regression Head (optional, can be removed if only assignment is needed)

        per_lepton_repeated_global = keras.layers.RepeatVector(self.NUM_LEPTONS)(
            keras.layers.Flatten()(global_outputs)
        )

        jet_attention = assignment_probs
        if stop_gradient_assignment_probs:
            jet_attention = StopGradientLayer(name="stop_gradient_assignment")(
                assignment_probs
            )
        assosciated_jets = keras.layers.Dot(axes=(1, 1))([jet_attention, jet_outputs])
        regression_inputs = keras.layers.Concatenate(axis=-1)(
            [assosciated_jets, per_lepton_repeated_global]
        )

        regression_outputs = MLP(
            output_dim=3,
            name="normalized_regression",
            num_layers=3,
        )(regression_inputs)

        self._build_model_base(
            jet_assignment_probs=assignment_probs,
            regression_output=regression_outputs,
            name="CompactReconstructorModel",
        )
