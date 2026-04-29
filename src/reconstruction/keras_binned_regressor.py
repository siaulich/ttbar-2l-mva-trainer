import tensorflow as tf
import keras as keras
import numpy as np
from abc import ABC, abstractmethod
from ..preprocessing.training_data_loader import DataConfig
from ..base_classes import KerasMLWrapper
from ..components import UnbinRegressionOutput
from ..utils import losses
from ..base_classes.reconstruction_base import EventReconstructorBase

from copy import deepcopy


class KerasBinnedRegressor(EventReconstructorBase, KerasMLWrapper):
    def __init__(
        self,
        config: DataConfig,
        name,
        perform_regression=False,
        use_nu_flows=True,
        load_model_path=None,
    ):
        EventReconstructorBase.__init__(
            self,
            config=config,
            assignment_name=name,
            full_reco_name=(
                name
                if perform_regression
                else name + (r" + $\nu^2$-Flows" if use_nu_flows else r" + True $\nu$")
            ),
            neutrino_name=name,
            perform_regression=perform_regression,
            use_nu_flows=use_nu_flows,
        )
        KerasMLWrapper.__init__(
            self, config=config, perform_regression=perform_regression
        )
        if load_model_path is not None:
            self.load_model(load_model_path)
        self.predict_confidence = False

    def _build_model_base(
        self,
        jet_assignment_probs,
        regression_output,
        **kwargs,
    ):
        trainable_outputs = {}
        outputs = {}
        outputs["assignment"] = jet_assignment_probs
        trainable_outputs["assignment"] = jet_assignment_probs

        if regression_output is None:
            raise ValueError(
                "perform_regression is True but no regression_output provided to build_model."
            )
        outputs["binned_regression"] = regression_output
        trainable_outputs["binned_regression"] = regression_output
        self.model = keras.models.Model(
            inputs=self.inputs,
            outputs=outputs,
            name=kwargs.get("name", "reco_model"),
        )
        self.trainable_model = keras.models.Model(
            inputs=self.inputs,
            outputs=trainable_outputs,
            name=kwargs.get("name", "reco_trainable_model"),
        )

    def prepare_labels(self, X, y=None, copy_data=True):
        y_train = {}

        if y is None:
            y_train["assignment"] = X["assignment"]
            y_train["regression"] = X["regression"]
        else:
            y_train = y.copy() if not copy_data else deepcopy(y)

        regression_data = y_train.pop("regression")
        upscale_layer = self.model.get_layer("regression")
        if upscale_layer is None:
            raise ValueError(
                "Regression layer not found in model. Cannot prepare regression targets."
            )
        y_train["binned_regression"] = self.model.get_layer("regression").bin_data(
            regression_data
        )

        return y_train

    def compile_model(
        self, loss, optimizer, metrics=None, add_physics_informed_loss=False, **kwargs
    ):
        if self.trainable_model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before compile_model()."
            )
        self.trainable_model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics, **kwargs
        )

    def generate_one_hot_encoding(self, predictions, exclusive):
        prediction_product_matrix = (
            predictions[..., 0][:, :, np.newaxis]
            + predictions[..., 1][:, np.newaxis, ...]
        )  # shape (batch_size, max_jets, max_jets)
        if exclusive:
            prediction_product_matrix[
                :, np.arange(predictions.shape[1]), np.arange(predictions.shape[1])
            ] = 0  # set diagonal to zero to enforce exclusivity
        one_hot = np.zeros(
            (predictions.shape[0], self.max_jets, self.NUM_LEPTONS), dtype=int
        )
        idx = np.argmax(
            prediction_product_matrix.reshape(predictions.shape[0], -1), axis=1
        )
        one_hot[
            np.arange(predictions.shape[0]),
            np.unravel_index(idx, prediction_product_matrix.shape[1:])[0],
            0,
        ] = 1
        one_hot[
            np.arange(predictions.shape[0]),
            np.unravel_index(idx, prediction_product_matrix.shape[1:])[1],
            1,
        ] = 1

        return one_hot

    def predict_indices(self, data: dict[str : np.ndarray], exclusive=True):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        return self.complete_forward_pass(data)[0]

    def reconstruct_neutrinos(self, data: dict[str : np.ndarray]):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        return self.complete_forward_pass(data)[1]

    def predict(self, data: dict[str : np.ndarray], batch_size=2048, verbose=0):
        inputs = {}
        for key in self.model.input:
            if hasattr(key, "name"):
                input_name = key.name.split(":")[0]
            elif isinstance(key, str):
                input_name = key
            else:
                raise ValueError(
                    f"Unexpected input key type: {type(key)}. Expected a Keras tensor or string."
                )
            if input_name in data:
                inputs[input_name] = data[input_name]
            else:
                raise ValueError(
                    f"Expected input '{input_name}' not found in data dictionary."
                )
        predictions = self.model.predict(inputs, batch_size=batch_size, verbose=verbose)
        return predictions

    def complete_forward_pass(self, data: dict[str : np.ndarray]):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        predictions = self.predict(data)
        assignment_predictions = self.generate_one_hot_encoding(
            predictions["assignment"], exclusive=True
        )
        if "regression" in predictions:
            neutrino_reconstruction = predictions["regression"]
        else:
            neutrino_reconstruction = EventReconstructorBase.reconstruct_neutrinos(
                self, data
            )
        return assignment_predictions, neutrino_reconstruction

    def adapt_output_layer_scales(self, data):
        regression_scale = np.percentile(np.abs(data["regression"]), 99.5, axis=0)
        print(f"Computed regression scale from data: {regression_scale}")
        unbinning_layer = UnbinRegressionOutput(
            scale=2 * regression_scale, name="regression"
        )
        outputs = self.model.output
        outputs["regression"] = unbinning_layer(
            self.model.get_layer("binned_regression").output
        )
        outputs.pop("binned_regression")
        self.model = keras.models.Model(
            inputs=self.model.inputs,
            outputs=outputs,
        )
        print("Set regression denormalization layer with computed mean and std.")

    def compute_sample_weights(self, data, **kwargs):
        one_hot_binned_regression = self.model.get_layer("regression").bin_data(
            data["regression"]
        )
        alpha = kwargs.get("alpha", 0.25)
        class_weights = np.sum(one_hot_binned_regression, axis=0) / np.sum(
            one_hot_binned_regression
        )
        class_weights = 1 / (
            class_weights + 1e-4
        )  # Invert to get higher weight for less frequent classes
        sample_weights = one_hot_binned_regression * class_weights[np.newaxis, :]
        sample_weights = np.prod(
            sample_weights, axis=tuple(range(1, sample_weights.ndim))
        )
        sample_weights = sample_weights / np.mean(
            sample_weights
        )  # Normalize to mean of 1
        sample_weights = (sample_weights) ** alpha  # Apply scaling factor
        return sample_weights
