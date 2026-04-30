import keras as keras
import tensorflow as tf
import numpy as np


from ..reconstruction import KerasNuPriorAssigner
from ..components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    ExpandJetMask,
    SplitTransformerOutput,
    JetLeptonAssignment,
    EmbeddingMLP,
)

from ..configs import DataConfig


class NuFlowsPriorAssigner(KerasNuPriorAssigner):
    def __init__(self, config: DataConfig, name="CrossAttentionModel"):
        super().__init__(config, name=name)
        self.perform_regression = False

    def build_model(
        self,
        hidden_dim,
        num_layers,
        num_heads=8,
        dropout_rate=0.1,
        log_variables=False,
        compute_HLF=False,
        use_global_event_inputs=True,
    ):
        """
        Builds a more compact version of the Assignment Transformer model with fewer parameters.
        """
        # Input layers
        normed_inputs, masks = self._prepare_inputs(
            log_variables=log_variables,
            compute_HLF=True,
            use_global_event_inputs=use_global_event_inputs,
        )

        normed_jet_inputs = normed_inputs["jet_inputs"]
        normed_lep_inputs = normed_inputs["lepton_inputs"]
        normed_neutrino_inputs = normed_inputs["nu_flows_neutrino_regression"]
        normed_global_event_inputs = normed_inputs["global_event_inputs"]
        normed_met_inputs = normed_inputs["met_inputs"]
        jet_mask = masks["jet_mask"]

        # Concatenate global event features with MET inputs
        global_event_features = keras.layers.Concatenate(axis=-1)(
            [normed_global_event_inputs, normed_met_inputs]
        )

        normed_lep_inputs = keras.layers.Concatenate(axis=-1)(
            [normed_lep_inputs, normed_neutrino_inputs]
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
        )(global_event_features)

        # Concatenate objects to sequence
        sequence = keras.layers.Concatenate(axis=1)(
            [jet_embedding, lep_embedding, global_embedding]
        )
        sequence_mask = ExpandJetMask(
            name="expand_jet_mask",
            extra_sequence_length=self.NUM_LEPTONS
            + 1,  # leptons + neutrino priors + global features
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
        jet_outputs, lepton_outputs, _ = SplitTransformerOutput(
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
        self._build_model_base(
            assignment_logits,
        )
