import numpy as np
import matplotlib.pyplot as plt
import atlas_mpl_style as ampl
import tensorflow as tf
import keras

ampl.use_atlas_style()
ampl.set_color_cycle("ATLAS")
from typing import Optional, Union, List, Dict
from copy import deepcopy

from ..base_classes import KerasMLWrapper
import time
import os
from ..components import MultiHeadAttentionBlock, SelfAttentionBlock
from ..utils import prepare_model_inputs


def build_latent_space_extractor(
    model: keras.Model, layer_names: Union[str, List[str]]
) -> keras.Model:
    """
    Builds a model that returns both the original outputs
    and the activations from specified layers.

    Args:
        model (keras.Model): The Keras model for which to build the extractor.
        layer_names (str or list of str, optional): Names of layers to extract activations from.

    Returns:
        keras.Model: A new model that outputs both original outputs and specified layer activations.
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    inputs = model.input
    activations = {}

    for layer in model.layers:
        if layer.name in layer_names:
            activations[layer.name + "_latent"] = layer.output

    extractor = keras.Model(inputs=inputs, outputs={**activations})
    return extractor


def get_pre_layer_model(model: keras.Model, layer_name: str) -> keras.Model:
    """
    Builds a model that outputs the activations from the layers immediately preceding the specified layer.

    Args:
        model (keras.Model): The Keras model for which to build the extractor.
        layer_name (str): Name of the layer for which to extract the preceding layer's activations.

    Returns:
        keras.Model: A new model that outputs the activations from the preceding layers.
    """
    inputs = model.input
    layer = model.get_layer(layer_name)
    if layer.input is None:
        raise ValueError(f"Layer '{layer_name}' does not have an input.")
    outputs = layer.input
    pre_layer_model = keras.Model(
        inputs=inputs, outputs=outputs, name=f"pre_{layer_name}_model"
    )
    return pre_layer_model


def extract_attention_scores(
    model: keras.Model, X: dict, layer_names: Union[str, List[str]]
) -> Dict[str, np.ndarray]:
    """
    Extracts attention scores from specified layers in the model.

    Args:
        model (keras.Model): The Keras model from which to extract attention scores.
        X (dict): Input data for the model.
        layer_names (str or list of str): Names of layers to extract attention scores from.

    Returns:
        dict: A dictionary mapping layer names to their corresponding attention scores.
    """
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    attention_scores = {}
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            raise ValueError(f"Layer '{layer_name}' not found in model '{model.name}'.")
        pre_layer_model = get_pre_layer_model(model, layer_name)
        pre_layer_outputs = pre_layer_model.predict(
            prepare_model_inputs(pre_layer_model, X)
        )
        if isinstance(pre_layer_outputs, np.ndarray):
            output = layer(pre_layer_outputs, return_attention_scores=True)
        else:
            output = layer(*pre_layer_outputs, return_attention_scores=True)

        if isinstance(output, tuple) and len(output) == 2:
            _, attn_scores = output
            attention_scores[layer_name] = attn_scores.numpy()
        else:
            print(
                f"WARNING: Layer '{layer_name}' does not return attention scores."
            )
    return attention_scores
