import keras as keras
from ..reconstruction import KerasFFGaussian
from .. import DataConfig
from ..components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    ExpandJetMask,
    SplitTransformerOutput,
    JetLeptonAssignment,
    EmbeddingMLP,
    StopGradientLayer,
)

class CompactReconstructorVariance(KerasFFGaussian):
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
        assignment_probs = JetLeptonAssignment(
            dim=hidden_dim, name="assignment"
        )(
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
            jet_attention = StopGradientLayer(
                name="stop_gradient_assignment"
            )(assignment_probs)
        assosciated_jets = keras.layers.Dot(axes=(1, 1))([jet_attention, jet_outputs])
        regression_inputs = keras.layers.Concatenate(axis=-1)(
            [assosciated_jets, per_lepton_repeated_global]
        )

        regression_mean = MLP(
            output_dim=3,  # variance for all three regression targets
            name="regression_mean",
            num_layers=3,
        )(regression_inputs)

        regression_var = MLP(
            output_dim=3,  # variance for all three regression targets
            activation="softplus",  # ensure positivity of variance
            name="regression_var",
            num_layers=3,
        )(regression_inputs)

        regression_outputs = keras.layers.Concatenate(
            name="normalized_regression", axis=-1
        )([regression_mean, regression_var])

        self._build_model_base(
            jet_assignment_probs=assignment_probs,
            regression_output=regression_outputs,
            name="CompactReconstructorModel",
        )
