import sys
import argparse
import os
import numpy as np
import yaml
import pandas as pd
from dacite import from_dict, Config
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from copy import deepcopy

plt.rcParams["font.size"] = 18
import matplotlib as mpl

mpl.rcParams["figure.constrained_layout.use"] = True
from core.DataLoader import DataPreprocessor
from core import evaluation, reconstruction, base_classes, keras_models

from core.configs import (
    EvaluationConfig,
    DataConfig,
    LoadConfig,
    load_yaml_config,
    load_hyperparameter_evaluation_config,
    get_load_config_from_yaml,
)


def parse_args():
    """Parse command line arguments for running the hyperparameter evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with different hyperparameters on specified datasets"
    )

    # Configuration file arguments
    parser.add_argument(
        "--load_config",
        type=str,
        required=True,
        help="Path to the load configuration YAML file",
    )
    parser.add_argument(
        "--evaluation_config",
        type=str,
        required=True,
        help="Path to the evaluation configuration YAML file",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=2_000_000,
        help="Number of events to evaluate (default: 2,000,000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/hyperparameter_evaluation_results",
        help="Directory to save evaluation results (default: ./hyperparameter_evaluation_results)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configurations
    load_config = get_load_config_from_yaml(args.load_config)
    hyperparameter_config = load_hyperparameter_evaluation_config(
        args.evaluation_config
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model_type = keras_models._get_model(hyperparameter_config.model_architecture.type)
    hyper_parameter_shape = tuple(
        (
            len(hyperparameter.values)
            for hyperparameter in hyperparameter_config.hyperparameters
        )
    )
    print(
        f"Evaluating {np.prod(hyper_parameter_shape)} models with different hyperparameter combinations..."
    )

    model_data_frame = pd.DataFrame()
    index_names = [hp.name for hp in hyperparameter_config.hyperparameters]
    index_arrays = [hp.values for hp in hyperparameter_config.hyperparameters]
    multi_index = pd.MultiIndex.from_product(index_arrays, names=index_names)

    # Create dataframe with multi-index and models column
    model_data_frame = pd.DataFrame(index=multi_index)
    assignment_accuracy= pd.Series(index=model_data_frame.index, dtype=float)
    regression_mse = pd.Series(index=model_data_frame.index, dtype=float)
    num_trainable_parameters = pd.Series(index=model_data_frame.index, dtype=int)
    inference_time_per_event = pd.Series(index=model_data_frame.index, dtype=float)
    training_epochs = pd.Series(index=model_data_frame.index, dtype=int)

    # Load data
    data_processor = DataPreprocessor(load_config)
    data_config = data_processor.load_from_npz(
        npz_path=load_config.data_path,
        max_events=args.num_events,
        event_numbers=hyperparameter_config.evaluation_event_numbers,
    )
    X, y = data_processor.get_data()
    num_events = data_processor.get_num_events()
    del data_processor
    print(f"Loaded {num_events} events for evaluation.")

    for hyperparameter_combination in multi_index:

        print(
            f"Evaluating model with hyperparameters: {dict(zip(index_names, hyperparameter_combination))}"
        )
        file_name = hyperparameter_config.model_architecture.file_name_pattern.format(
            *hyperparameter_combination
        )
        if not os.path.exists(file_name):
            print(f"Model file {file_name} does not exist. Skipping this combination.")
            continue
        options = deepcopy(hyperparameter_config.model_architecture.options)
        model = model_type(data_config, **options)
        model.load_model(file_name)
        print(f"Loaded model from {file_name} for evaluation.")
        ml_evaluator = evaluation.MLEvaluator([model], X, y)
        evaluation_results = ml_evaluator.evaluate_idx(0)
        assignment_accuracy.loc[hyperparameter_combination] = evaluation_results["assignment_accuracy"]
        regression_mse.loc[hyperparameter_combination] = evaluation_results["regression_mse"]
        num_trainable_parameters.loc[hyperparameter_combination] = ml_evaluator.evaluate_num_parameters_idx(0)["num_trainable_parameters"]
        inference_time_per_event.loc[hyperparameter_combination] = ml_evaluator.evaluate_inference_time_idx(0)["time_per_sample"]
        training_epochs.loc[hyperparameter_combination] = ml_evaluator.evaluate_num_training_epochs_idx(0)["num_training_epochs"]
        del ml_evaluator
        del model
    
    model_data_frame["assignment_accuracy"] = assignment_accuracy
    model_data_frame["regression_mse"] = regression_mse
    model_data_frame["num_trainable_parameters"] = num_trainable_parameters
    model_data_frame["inference_time_per_event"] = inference_time_per_event
    model_data_frame["training_epochs"] = training_epochs
    # Save evaluation results to CSV
    output_csv_path = os.path.join(args.output_dir, "hyperparameter_evaluation_results.csv")
    model_data_frame.to_csv(output_csv_path)
    print(f"Saved hyperparameter evaluation results to {output_csv_path}.")
    