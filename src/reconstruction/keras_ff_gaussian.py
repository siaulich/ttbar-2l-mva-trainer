import tensorflow as tf
import keras as keras
import numpy as np
from abc import ABC, abstractmethod
from ..configs import DataConfig
from ..base_classes import KerasMLWrapper
from ..components import (
    OutputUpScaleLayer,
    PhysicsInformedLoss,
    ConfidenceLossOutputLayer,
)
from ..utils import losses
from ..base_classes.reconstruction_base import EventReconstructorBase

from copy import deepcopy


class KerasFFGaussian(EventReconstructorBase, KerasMLWrapper):
    def __init__(
        self,
        config: DataConfig,
        name,
        perform_regression=False,
        use_nu_flows=True,
        use_mean=True,
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
        self.use_mean = use_mean

    def _build_model_base(
        self,
        jet_assignment_probs,
        regression_output=None,
        confidence_score=None,
        **kwargs,
    ):
        trainable_outputs = {}
        outputs = {}
        outputs["assignment"] = jet_assignment_probs
        trainable_outputs["assignment"] = jet_assignment_probs

        if self.perform_regression and regression_output is None:
            raise ValueError(
                "perform_regression is True but no regression_output provided to build_model."
            )
        if self.perform_regression and regression_output is not None:
            outputs["normalized_regression"] = regression_output
            trainable_outputs["normalized_regression"] = regression_output
        if confidence_score is not None:
            outputs["confidence_score"] = confidence_score
            trainable_outputs["confidence_loss_output"] = ConfidenceLossOutputLayer(
                name="confidence_loss_output"
            )(jet_assignment_probs, confidence_score)
            self.predict_confidence = True
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

    def add_reco_mass_deviation_loss(self):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before adding physics-informed loss. If you use regression, adapt the normalization layers first."
            )
        if "regression" not in self.model.output_names:
            raise ValueError(
                "Regression output not found in model outputs. Cannot add physics-informed loss."
            )
        reco_mass_deviation_layer = PhysicsInformedLoss(name="reco_mass_deviation")
        neutrino_momenta = self.model.get_layer("regression").output

        if neutrino_momenta is None:
            raise ValueError(
                "Regression output not found in model outputs. Cannot add physics-informed loss."
            )

        lepton_momenta = self.inputs["lep_inputs"]

        reco_mass_deviation = reco_mass_deviation_layer(
            neutrino_momenta, lepton_momenta
        )
        trainable_outputs = {**self.trainable_model.output}
        trainable_outputs["reco_mass_deviation"] = reco_mass_deviation
        self.trainable_model = keras.Model(
            inputs=self.model.inputs,
            outputs=trainable_outputs,
        )

        print("Added physics-informed loss to the model.")

    def prepare_labels(self, X, y=None, copy_data=True):
        y_train = {}

        if y is None:
            y_train["assignment"] = X["assignment"]
            y_train["regression"] = X["regression"] if "regression" in y else None
        else:
            y_train = y.copy() if not copy_data else deepcopy(y)

        if not self.perform_regression:
            y_train.pop("regression")

        if "regression" in y_train:
            regression_data = y_train.pop("regression")
            upscale_layer = self.model.get_layer("regression")
            if upscale_layer is None:
                raise ValueError(
                    "Regression layer not found in model. Cannot prepare regression targets."
                )
            if not isinstance(upscale_layer, keras.layers.Rescaling):
                raise ValueError(
                    "Regression layer is not a Rescaling layer. Cannot prepare regression targets."
                )
            regression_std = upscale_layer.scale[
                ..., :3
            ]  # Assuming the first 3 entries correspond to the regression output
            if isinstance(regression_data, dict):
                y_train["normalized_regression"] = {}
                for key in regression_data:
                    if regression_data[key] is not None:
                        y_train["normalized_regression"][key] = (
                            regression_data[key] / regression_std
                        )
                    else:
                        y_train["normalized_regression"][key] = None
            else:
                y_train["normalized_regression"] = regression_data / regression_std

        if "reco_mass_deviation" in self.trainable_model.output:
            y_train["reco_mass_deviation"] = np.zeros(
                (y_train["assignment"].shape[0], 1), dtype=np.float32
            )
        if "confidence_loss_output" in self.trainable_model.output:
            y_train["confidence_loss_output"] = y_train["assignment"]
        return y_train

    def compile_model(
        self, loss, optimizer, metrics=None, add_physics_informed_loss=False, **kwargs
    ):
        if self.trainable_model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before compile_model()."
            )
        if self.predict_confidence:
            loss["confidence_loss_output"] = losses.ConfidenceScoreLoss()
        if add_physics_informed_loss:
            self.add_reco_mass_deviation_loss()
            print(
                "Compiling model with physics informed loss. Ensure that the loss dictionary includes 'reco_mass_deviation'."
            )
            if "reco_mass_deviation" not in loss:
                loss["reco_mass_deviation"] = lambda y_true, y_pred: y_pred
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
            if self.use_mean:
                neutrino_reconstruction = predictions["regression"][..., :3]
            else:
                neutrino_momenta_mean = predictions["regression"][..., :3]
                neutrino_momenta_var = predictions["regression"][..., 3:]
                neutrino_reconstruction = np.random.normal(
                    loc=neutrino_momenta_mean, scale=np.sqrt(neutrino_momenta_var)
                )
        else:
            print(
                "No regression output found in model predictions. Using default neutrino reconstruction."
            )
            neutrino_reconstruction = EventReconstructorBase.reconstruct_neutrinos(
                self, data
            )
        return assignment_predictions, neutrino_reconstruction

    def adapt_output_layer_scales(self, data):
        if (
            self.perform_regression
            and "normalized_regression" in self.model.output_names
        ):
            denormalisation_layer = keras.layers.Rescaling(
                scale=np.array(
                    [1e5, 1e5, 1e5, 1e10, 1e10, 1e10]
                ),  # Scale neutrinos to 100 GeV by default
                name="regression",
            )
            outputs = self.model.output
            outputs["regression"] = denormalisation_layer(
                self.model.get_layer("normalized_regression").output
            )
            outputs.pop("normalized_regression")
            self.model = keras.models.Model(
                inputs=self.model.inputs,
                outputs=outputs,
            )
        else:
            print(
                "No regression output found in model outputs to adapt scales for. Skipping output layer scale adaptation."
            )
