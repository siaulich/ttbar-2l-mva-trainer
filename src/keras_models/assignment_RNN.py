import keras as keras
import tensorflow as tf
from keras.layers import LSTM, GRU, RNN, LSTMCell, Bidirectional, GRUCell

from ..reconstruction import KerasFFRecoBase
from ..configs import DataConfig
from ..components import (
    MLP,
    TemporalSoftmax,
)


class FeatureConcatRNN(KerasFFRecoBase):
    def __init__(self, config: DataConfig, name="RNN"):
        if config.has_neutrino_truth:
            print(
                "FeatureConcatRNN is designed for classification tasks; regression targets will be ignored."
            )
        super().__init__(config, assignment_name=name, perform_regression=False)

    def build_model(self, hidden_dim, num_layers, dropout_rate, recurrent_type="lstm"):

        # Input layers
        jet_inputs, lep_inputs, met_inputs, jet_mask = self._prepare_inputs()

        flatted_met_inputs = keras.layers.Flatten()(met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(lep_inputs)

        # Concat lepton and met features to jets
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_inputs = keras.layers.Concatenate(axis=-1)(
            [jet_inputs, met_repeated_jets, lepton_repeated_jets]
        )

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_inputs)

        # Recurrent layers
        if recurrent_type.lower() == "lstm":
            RNNCell = LSTMCell
        elif recurrent_type.lower() == "gru":
            RNNCell = GRUCell
        else:
            raise ValueError("recurrent_type must be either 'lstm' or 'gru'")

        x = jet_embedding
        for i in range(num_layers):
            x = keras.layers.Bidirectional(
                RNN(
                    RNNCell(
                        units=hidden_dim,
                        dropout=dropout_rate,
                    ),
                    return_sequences=True,
                ),
                name=f"bidir_rnn_{i}",
            )(x, mask=jet_mask)

            x = keras.layers.LayerNormalization(name=f"rnn_ln_{i}")(x)

        # Output layers
        jet_assignment_probs = MLP(
            self.NUM_LEPTONS,
            num_layers=3,
            name="jet_assignment_mlp",
        )(x)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_assignment_probs, mask=jet_mask
        )
        # Build model
        self._build_model_base(jet_assignment_probs, name="FeatureConcatRNN")
