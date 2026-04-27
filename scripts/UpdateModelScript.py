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
from src.DataLoader import (
    DataPreprocessor,
    DataConfig,
    LoadConfig,
)
from src.configs import ModelConfig, TrainConfig, load_yaml_config


def parse_args():
    """Parse Command line arguments for running the updating script."""

    parser = argparse.ArgumentParser(
        description="Update a pre-trained model with new data and specified hyperparameters"
    )

    # Configuration file arguments
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the pre-trained model and its configuration files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load pre-trained model and its configuration
    model_dir = args.model_dir
    config = load_yaml_config(os.path.join(model_dir, "model_config.yaml"))

    model_config = ModelConfig(**config["ModelConfig"])
    train_config = TrainConfig(**config["TrainConfig"])
    load_config = LoadConfig(**config["LoadConfig"])
    data_config = load_config.to_data_config()
    # Load the pre-trained model
    for file in os.listdir(model_dir):
        if file.endswith(".keras"):
            model_path = os.path.join(model_dir, file)
            keras_model = keras_models._get_model(model_config.model_type)(data_config)
            keras_model.load_model(model_path)
            keras_model.export_to_onnx(
                os.path.join(model_dir, file.replace(".keras", ".onnx"))
            )
            print(
                f"Updated model loaded from {model_path} and exported to ONNX format."
            )
    print("Model update process completed successfully.")
