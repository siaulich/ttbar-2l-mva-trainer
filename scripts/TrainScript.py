import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from copy import deepcopy

from src import keras_models, utils
from src.preprocessing.training_data_loader import (
    TrainingDataLoader,
    DataConfig,
    LoadConfig,
)

from src.configs import ModelConfig, TrainConfig, load_yaml_config


def parse_args():
    """Parse command line arguments for running the training script."""
    parser = argparse.ArgumentParser(
        description="Train Transformer model with specified hyperparameters"
    )

    # Configuration file arguments
    parser.add_argument(
        "--load_config",
        type=str,
        required=True,
        help="Path to the load configuration YAML file",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save trained models and logs",
    )
    parser.add_argument(
        "--event_numbers",
        type=str,
        default="all",
        help="Comma-separated list of event numbers to use for training (optional)",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=None,
        help="Maximum number of events to load from the dataset (optional)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_config = LoadConfig(**load_yaml_config(args.load_config)["LoadConfig"])
    train_config = TrainConfig(**load_yaml_config(args.train_config)["TrainConfig"])
    model_config = ModelConfig(**load_yaml_config(args.model_config)["ModelConfig"])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    num_events = None
    if model_config.num_events is not None:
        num_events = model_config.num_events
    if args.num_events is not None:
        num_events = args.num_events

    data_preprocessor = TrainingDataLoader(load_config)
    data_config = data_preprocessor.load_from_npz(
        load_config.data_path,
        event_numbers=args.event_numbers,
        num_events=num_events,
    )
    X, y = data_preprocessor.get_data()
    num_events = X["jet_inputs"].shape[0]

    print(f"Data loaded. Using {num_events} events for training.")

    model = keras_models._get_model(model_config.model_type)(data_config)

    build_options = deepcopy(model_config.model_options)
    build_options.update(**model_config.model_params)

    model.build_model(**(build_options))

    model.adapt_normalization_layers(X)

    compile_options = deepcopy(model_config.compile_options)

    losses = {
        key: getattr(utils, value["class_name"])(**value.get("config", {}))
        for key, value in compile_options["loss"].items()
    }
    compile_options.pop("loss", None)
    metrics = {
        key: [
            getattr(utils, metric["class_name"])(**metric.get("config", {}))
            for metric in value
        ]
        for key, value in compile_options["metrics"].items()
    }
    compile_options.pop("metrics", None)
    optimizer = keras.optimizers.get(compile_options["optimizer"])
    compile_options.pop("optimizer", None)
    model.compile_model(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        **compile_options,
    )
    compile_options.update(optimizer=optimizer, loss=losses, metrics=metrics)

    train_options = deepcopy(train_config.__dict__)

    callbacks = []
    for callback_name, callback_params in train_config.callbacks.items():
        callback_class = getattr(keras.callbacks, callback_name)
        callbacks.append(callback_class(**callback_params))
    train_options["callbacks"] = callbacks

    X_train, y_train, sample_weights = model.prepare_training_data(X, y)
    del X, y  # Free memory

    even_history = model.train_model(
        X_train,
        y_train,
        sample_weight=sample_weights,
        **train_options,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.event_numbers == "even":
        model.save_model(os.path.join(args.output_dir, f"odd_model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "odd_model.onnx"))
    elif args.event_numbers == "odd":
        model.save_model(os.path.join(args.output_dir, f"even_model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "even_model.onnx"))
    else:
        model.save_model(os.path.join(args.output_dir, f"model.keras"))
        model.export_to_onnx(os.path.join(args.output_dir, "model.onnx"))

    with open(os.path.join(args.output_dir, "model_config.yaml"), "w") as file:
        yaml.dump(
            {
                "LoadConfig": load_config.__dict__,
                "TrainConfig": train_config.__dict__,
                "ModelConfig": model_config.__dict__,
            },
            file,
        )
