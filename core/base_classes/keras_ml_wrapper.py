from abc import ABC, abstractmethod
from . import BaseUtilityModel
from core.DataLoader import DataConfig
import keras as keras
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import os
from core.components import (
    onnx_support,
    GenerateMask,
    ComputeHighLevelFeatures_from_PtEtaPhiE,
    InputMetLayer,
    ProcessPtEtaPhiELayer,
    PhysicsInformedLoss,
)
import core.components as components
import core.utils as utils
from copy import deepcopy


@keras.utils.register_keras_serializable()
class KerasModelWrapper(keras.Model):
    def predict_dict(self, x, batch_size=2048, verbose=0, steps=None, **kwargs):
        model_input_data = {}
        for input in self.input:
            input_name = input.name
            if input_name not in x:
                raise ValueError(
                    f"Model expects input '{input_name}' which is not present in the data."
                )
            model_input_data[input_name] = x[input_name]
        predictions = super().predict(
            model_input_data,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            **kwargs,
        )
        if not isinstance(predictions, dict):
            if not isinstance(predictions, list):
                predictions = [predictions]
            return dict(zip(self.output_names, predictions))
        return predictions


class KerasMLWrapper(BaseUtilityModel, ABC):
    def __init__(
        self,
        config: DataConfig,
        perform_regression=False,
        name=None,
        **kwargs,
    ):
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        """
        self.model: keras.models.Model = None
        self.trainable_model: keras.models.Model = None
        self.history = None
        self.NUM_LEPTONS = config.NUM_LEPTONS
        self.max_jets = config.max_jets
        self.met_features = config.met_features
        self.n_jets: int = len(config.jet_features)
        self.n_leptons: int = len(config.lepton_features)
        self.n_met: int = len(config.met_features) if config.met_features else 0
        self.n_global: int = (
            len(config.global_event_inputs) if config.has_global_event_inputs else 0
        )
        self.padding_value: float = config.padding_value
        self.feature_index_dict = config.feature_indices

        # initialize empty dicts to hold inputs and transformed inputs
        self.inputs = {}
        self.transformed_inputs = {}
        self.normed_inputs = {}
        self.masks = {}
        self.model_id = None
        self.perform_regression = perform_regression


    def build_model(self, **kwargs):
        raise NotImplementedError("Subclasses must implement build_model method.")
        pass

    def prepare_training_data(
        self, X,y =None, sample_weights=None, class_weights=None, copy_data=False
    ):
        for input_name in self.inputs.keys():
            if input_name not in X:
                raise ValueError(f"Input '{input_name}' not found in data dictionary.")

        if copy_data:
            X_train = {key: deepcopy(X[key]) for key in self.inputs.keys()}
        else:
            X_train = {key: X[key] for key in self.inputs.keys()}

        y_train = self.prepare_labels(X, y)
        
        return X_train, y_train, sample_weights

    def prepare_labels(self, X,y):
        return y

    def _prepare_inputs(
        self, log_variables=True, compute_HLF=False, use_global_event_inputs=False
    ):
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(
            shape=(self.NUM_LEPTONS, self.n_leptons), name="lep_inputs"
        )
        met_inputs = keras.Input(shape=(1, self.n_met), name="met_inputs")

        if self.config.has_global_event_inputs:
            global_event_inputs = keras.Input(
                shape=(1, self.n_global,),
                name="global_event_inputs",
            )

        # Generate jet mask
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)

        # Transform inputs
        transformed_jet_inputs = ProcessPtEtaPhiELayer(
            name="jet_input_transform",
            padding_value=self.padding_value,
            log_variables=log_variables,
        )(jet_inputs)
        transformed_lep_inputs = ProcessPtEtaPhiELayer(
            name="lep_input_transform",
            padding_value=self.padding_value,
            log_variables=log_variables,
        )(lep_inputs)
        transformed_met_inputs = InputMetLayer(name="met_input_transform", log_variables=log_variables)(met_inputs)

        if compute_HLF:
            high_level_features = ComputeHighLevelFeatures_from_PtEtaPhiE(
                name="compute_high_level_features",
                padding_value=self.padding_value,
            )(
                jet_input=jet_inputs,
                lepton_input=lep_inputs,
                jet_mask=jet_mask,
            )

        normed_jet_inputs = keras.layers.Normalization(name="jet_input_normalization")(
            transformed_jet_inputs
        )
        normed_lep_inputs = keras.layers.Normalization(name="lep_input_normalization")(
            transformed_lep_inputs
        )
        normed_met_inputs = keras.layers.Normalization(name="met_input_normalization")(
            transformed_met_inputs
        )

        if self.config.has_global_event_inputs and use_global_event_inputs:
            print("Adding normalization for global event features")
            normed_global_event_inputs = keras.layers.Normalization(
                name="global_event_input_normalization"
            )(global_event_inputs)
        self.inputs = {
            "jet_inputs": jet_inputs,
            "lep_inputs": lep_inputs,
            "met_inputs": met_inputs,
        }
        self.transformed_inputs = {
            "jet_inputs": transformed_jet_inputs,
            "lepton_inputs": transformed_lep_inputs,
            "met_inputs": transformed_met_inputs,
        }
        self.normed_inputs = {
            "jet_inputs": normed_jet_inputs,
            "lepton_inputs": normed_lep_inputs,
            "met_inputs": normed_met_inputs,
        }
        self.masks = {"jet_mask": jet_mask}

        if compute_HLF:
            normed_hlf_inputs = keras.layers.Normalization(
                name="hlf_input_normalization", axis=(-1, -2)
            )(high_level_features)
            self.transformed_inputs["hlf_inputs"] = high_level_features
            self.normed_inputs["hlf_inputs"] = normed_hlf_inputs

        if self.config.has_global_event_inputs and use_global_event_inputs:
            self.inputs["global_event_inputs"] = global_event_inputs
            self.transformed_inputs["global_event_inputs"] = global_event_inputs
            self.normed_inputs["global_event_inputs"] = normed_global_event_inputs

        return self.normed_inputs, self.masks

    def train_model(
        self,
        X,
        y,
        epochs,
        batch_size,
        sample_weight=None,
        validation_split=0.2,
        callbacks=[],
        **kwargs,
    ):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before train_model()."
            )

        if self.trainable_model is None:
            self.trainable_model = self.model

        callbacks = callbacks if callbacks is not None else []
        callbacks.append(
            keras.callbacks.TerminateOnNaN()
        )  # Ensure training stops on NaN loss

        if self.history is None:
            self.history = self.trainable_model.fit(
                X,
                y,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                **kwargs,
            )
        else:
            append_history = self.trainable_model.fit(
                X,
                y,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                **kwargs,
            )
            # Append new history to existing history
            for key in append_history.history:
                self.history.history[key].extend(append_history.history[key])
            self.history.epoch.extend(
                [e + self.history.epoch[-1] + 1 for e in append_history.epoch]
            )
        return self.history

    def save_model(self, file_path="model.keras"):
        """
        Saves the current model to the specified file path in Keras format and writes the model's structure to a text file.
        Args:
            file_path (str): The file path where the model will be saved.
                            Must end with ".keras". Defaults to "model.keras".
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".keras" not in file_path:
            raise ValueError(
                "File path must end with .keras. Please provide a valid file path."
            )
        self.model.save(file_path)

        if self.history is not None:
            history_path = file_path.replace(".keras", "_history")
            np.savez(history_path, **self.history.history)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, model_id=None):
        """
        Loads a pre-trained Keras model from the specified file path.

        This method uses custom objects defined in the `CustomObjects` class to
        ensure that any custom layers, loss functions, or metrics used in the
        model are properly loaded.

        Args:
            file_path (str): The file path to the saved Keras model.
        """

        custom_objects = {
            name: obj
            for name, obj in zip(components.__dict__.items(), utils.__dict__.items())
            if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
        }
        custom_objects.update({"keras.models.Model": keras.models.Model})
        self.model_id = model_id

        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")
        history_path = file_path.replace(".keras", "_history.npz")
        if os.path.exists(history_path):
            loaded_history = np.load(history_path, allow_pickle=True)
            history_dict = {key: loaded_history[key].tolist() for key in loaded_history}
            self.history = keras.callbacks.History()
            self.history.history = history_dict
            print(f"Training history loaded from {history_path}")
        else:
            print(f"WARNING: No training history found at {history_path}")

    def adapt_normalization_layers(self, data: dict):
        """
        Adapts the normalization layers in a functional model.
        Each normalization layer is adapted using data that has been passed
        through all preceding layers (up to but excluding the normalization layer).
        """
        # --- Prepare and unpad jet data ---
        for key in self.inputs.keys():
            if key not in data:
                raise ValueError(f"Expected input '{key}' not found in data dictionary.")


        jet_data = data["jet_inputs"]  # (num_events, n_jets, n_features)
        jet_mask = np.any(jet_data != self.padding_value, axis=-1)
        unpadded_jet_data = jet_data[jet_mask]
        num_jets = unpadded_jet_data.shape[0]
        num_events = num_jets // self.max_jets
        unpadded_jet_data = unpadded_jet_data[: num_events * self.max_jets, :].reshape(
            (num_events, self.max_jets, self.n_jets)
        )
        lep_data = data["lep_inputs"][:num_events, :, :]
        met_data = data["met_inputs"][:num_events, :, :]
        global_event_data = None
        if "global_event_inputs" in data:
            global_event_data = data["global_event_inputs"][:num_events, :, :]

        # --- Helper: build a submodel up to (but not including) a target layer ---
        def get_pre_norm_submodel(model, target_layer_name):
            target_layer = model.get_layer(target_layer_name)
            # find which input(s) feed into this layer
            inbound_nodes = target_layer._inbound_nodes
            if not inbound_nodes:
                raise ValueError(f"Layer '{target_layer_name}' has no inbound nodes.")
            inbound_tensors = inbound_nodes[0].input_tensors
            submodel = keras.Model(
                inputs=model.inputs,
                outputs=(
                    inbound_tensors if len(inbound_tensors) > 1 else inbound_tensors[0]
                ),
                name=f"pre_{target_layer_name}_model",
            )
            return submodel

        input_data = {
            "jet_inputs": unpadded_jet_data,
            "lep_inputs": lep_data,
            "met_inputs": met_data,
        }
        if global_event_data is not None:
            input_data["global_event_inputs"] = global_event_data

        # --- Loop over normalization layers and adapt each ---
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Normalization):
                submodel = get_pre_norm_submodel(self.model, layer.name)
                submodel_inputs = submodel.inputs
                if not isinstance(submodel_inputs, list):
                    submodel_inputs = [submodel_inputs]

                submodel_input_data = {}
                for input_tensor in submodel_inputs:
                    input_name = input_tensor.name.split(":")[0]
                    if input_name not in input_data:
                        raise ValueError(
                            f"Input '{input_name}' required for submodel not found in input_data."
                        )
                    submodel_input_data[input_name] = input_data[input_name]
                # Get transformed data from submodel
                transformed_data = submodel.predict(
                    submodel_input_data,
                    batch_size=1024,
                    verbose=0,
                )
                layer.adapt(transformed_data)
                del submodel
                print("Adapted normalization layer: ", layer.name)

        self.adapt_output_layer_scales(data)
    
    def adapt_output_layer_scales(self, data: dict):
        print("No output layer scaling applied for this model.")
        pass

    def export_to_onnx(self, onnx_file_path="model.onnx"):
        """
        Exports the current Keras model to onnx format.

        The model is wrapped to take a flattened input tensor and split it into the original inputs.

        Saves a .txt file with model input names and positions for reference.

        Args:
            onnx_file_path (str): The file path where the ONNX model will be saved.
                                  Must end with ".onnx". Defaults to "model.onnx".
        Raises:
            ValueError: If the model has not been built (i.e., `self.model` is None).
            ValueError: If the provided file path does not end with ".onnx".
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".onnx" not in onnx_file_path:
            raise ValueError(
                "File path must end with .onnx. Please provide a valid file path."
            )

        input_shapes = {}
        for input in self.model.inputs:
            input_name = input.name.split(":")[0]
            input_shapes[input_name] = input.shape[1:]

        # Create a new model that takes a flat input and splits it
        flat_input_size = sum(np.prod(shape) for shape in input_shapes.values())
        flat_input = keras.Input(shape=(flat_input_size,), name="flat_input")

        split_layer = onnx_support.SplitInputsLayer(
            input_shapes.values(), name="split_inputs"
        )
        split_inputs = split_layer(flat_input)

        model_input_dict = {}
        for i, input in enumerate(input_shapes.keys()):
            model_input_dict[input] = split_inputs[i]

        model_outputs = self.model(model_input_dict)

        wrapped_model = keras.Model(inputs=flat_input, outputs=model_outputs)

        named_outputs = {}
        for k, v in model_outputs.items():
            named_outputs[k] = keras.layers.Lambda(
                lambda x, n=k: tf.identity(x, n), name=f"{k}"
            )(v)  # layer name irrelevant

        # Convert to ONNX
        spec = (tf.TensorSpec((None, flat_input_size), tf.float32, name="flat_input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(
            keras.models.Model(inputs=flat_input, outputs=named_outputs),
            input_signature=spec,
        )

        meta = onnx_model.metadata_props.add()
        meta.key = "NN_padding_value"
        meta.value = str(self.padding_value)

        meta = onnx_model.metadata_props.add()
        meta.key = "NN_max_jets"
        meta.value = str(self.max_jets)

        meta = onnx_model.metadata_props.add()
        meta.key = "input_names"
        meta.value = ",".join(input_shapes.keys())

        # Save ONNX model
        onnx.save_model(onnx_model, onnx_file_path)

        print(f"ONNX model saved to {onnx_file_path}")
        return wrapped_model

    def compute_sample_weights(self, X):
        return utils.compute_sample_weights(X)