import sys
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import numpy as np
import yaml
import pandas as pd
from copy import deepcopy
import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
import atlas_mpl_style as ampl

ampl.use_atlas_style()
ampl.set_color_cycle("ATLAS")
plt.rcParams["font.size"] = 18
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
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Whether to recompute evaluation results even if they already exist (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configurations
    load_config = get_load_config_from_yaml(args.load_config)
    evaluation_config = load_hyperparameter_evaluation_config(args.evaluation_config)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    models = dict()
    # Load data
    data_processor = DataPreprocessor(load_config)
    data_config = data_processor.load_from_npz(
        npz_path=load_config.data_path,
        max_events=args.num_events,
        event_numbers=evaluation_config.evaluation_event_numbers,
    )
    X, y = data_processor.get_data()
    num_events = data_processor.get_num_events()
    del data_processor
    print(f"Loaded {num_events} events for evaluation.")

    for hyperparameter_config in evaluation_config.models:
        index_names = [hp.name for hp in hyperparameter_config.hyperparameters]
        index_arrays = [hp.values for hp in hyperparameter_config.hyperparameters]
        multi_index = pd.MultiIndex.from_product(index_arrays, names=index_names)

        model_type = keras_models._get_model(hyperparameter_config.type)
        hyper_parameter_shape = tuple(
            (
                len(hyperparameter.values)
                for hyperparameter in hyperparameter_config.hyperparameters
            )
        )
        model_data_frame = None
        if not args.recompute:
            if os.path.exists(
                os.path.join(
                    args.output_dir,
                    f"{hyperparameter_config.name}_evaluation_results.csv",
                )
            ):
                model_data_frame = pd.read_csv(
                    os.path.join(
                        args.output_dir,
                        f"{hyperparameter_config.name}_evaluation_results.csv",
                    ),
                    index_col=index_names,
                )

        if model_data_frame is None:
            model_data_frame = pd.DataFrame(index=multi_index)
            assignment_accuracy = pd.Series(index=model_data_frame.index, dtype=float)
            regression_mse = pd.Series(index=model_data_frame.index, dtype=float)
            num_trainable_parameters = pd.Series(
                index=model_data_frame.index, dtype=int
            )
            inference_time_per_event = pd.Series(
                index=model_data_frame.index, dtype=float
            )
            training_epochs = pd.Series(index=model_data_frame.index, dtype=int)

            for hyperparameter_combination in tqdm.tqdm(
                multi_index, desc=f"Evaluating {hyperparameter_config.name}"
            ):

                file_name = hyperparameter_config.file_name_pattern.format(
                    *hyperparameter_combination
                )
                if not os.path.exists(file_name):
                    print(
                        f"Model file {file_name} does not exist. Skipping this combination."
                    )
                    continue
                options = deepcopy(hyperparameter_config.options)
                model = model_type(data_config, **options)
                model.load_model(file_name)
                ml_evaluator = evaluation.MLEvaluator([model], X, y)
                evaluation_results = ml_evaluator.evaluate_idx(0)
                assignment_accuracy.loc[hyperparameter_combination] = (
                    evaluation_results["assignment_accuracy"]
                )
                regression_mse.loc[hyperparameter_combination] = evaluation_results[
                    "regression_mse"
                ]
                num_trainable_parameters.loc[hyperparameter_combination] = (
                    ml_evaluator.evaluate_num_parameters_idx(0)[
                        "num_trainable_parameters"
                    ]
                )
                inference_time_per_event.loc[hyperparameter_combination] = (
                    ml_evaluator.evaluate_inference_time_idx(0)["time_per_sample"]
                )
                training_epochs.loc[hyperparameter_combination] = (
                    ml_evaluator.evaluate_num_training_epochs_idx(0)[
                        "num_training_epochs"
                    ]
                )
                del ml_evaluator
                del model

            model_data_frame["assignment_accuracy"] = assignment_accuracy
            model_data_frame["regression_mse"] = regression_mse
            model_data_frame["num_trainable_parameters"] = num_trainable_parameters
            model_data_frame["inference_time_per_event"] = inference_time_per_event
            model_data_frame["training_epochs"] = training_epochs
            # Save evaluation results to CSV
            output_csv_path = os.path.join(
                args.output_dir, f"{hyperparameter_config.name}_evaluation_results.csv"
            )
            model_data_frame.to_csv(output_csv_path)

        models[hyperparameter_config.name] = model_data_frame
    # Generate plots for evaluation results
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_data_frame in models.items():
        ax.scatter(
            model_data_frame["num_trainable_parameters"].values / 1e6,
            model_data_frame["assignment_accuracy"].values,
            label=model_name,
            alpha=0.7,
        )
    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0], xlims[1] * 1.1)
    ampl.set_xlabel(label="Number of Trainable Parameters [Mio]", ax=ax)
    ampl.set_ylabel(label="Assignment Accuracy", ax=ax)
    ampl.draw_atlas_label(x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress")
    ax.legend(loc="upper right")
    fig.savefig(
        os.path.join(args.output_dir, "assignment_accuracy_vs_num_parameters.pdf")
    )
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_data_frame in models.items():
        ax.scatter(
            model_data_frame["num_trainable_parameters"].values / 1e6,
            model_data_frame["training_epochs"].values,
            label=model_name,
            alpha=0.7,
        )
    ampl.set_xlabel(label="Number of Trainable Parameters [Mio]", ax=ax)
    ampl.set_ylabel(label="Number of Training Epochs", ax=ax)
    ampl.draw_atlas_label(x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(args.output_dir, "training_epochs_vs_num_parameters.pdf"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_data_frame in models.items():
        ax.scatter(
            model_data_frame["num_trainable_parameters"].values / 1e6,
            model_data_frame["inference_time_per_event"].values * 1e3,
            label=model_name,
            alpha=0.7,
        )
    ampl.set_xlabel(label="Number of Trainable Parameters [Mio]", ax=ax)
    ampl.set_ylabel(label="Inference Time per Event [ms]", ax=ax)
    ampl.draw_atlas_label(x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(args.output_dir, "inference_time_vs_num_parameters.pdf"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, model_data_frame in models.items():
        ax.scatter(
            model_data_frame["training_epochs"].values,
            model_data_frame["assignment_accuracy"].values,
            label=model_name,
            alpha=0.7,
        )
    ampl.set_xlabel(label="Number of Training Epochs", ax=ax)
    ampl.set_ylabel(label="Assignment Accuracy", ax=ax)
    ampl.draw_atlas_label(x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress")
    ax.legend(loc="upper right")
    fig.savefig(
        os.path.join(args.output_dir, "assignment_accuracy_vs_training_epochs.pdf")
    )
    plt.close(fig)

    for hyperparameter_config in evaluation_config.models:
        model_data_frame = models[hyperparameter_config.name]
        for plot_config in hyperparameter_config.plots_2d:
            x_variable = plot_config.x_variable
            y_variable = plot_config.y_variable
            if (
                x_variable not in model_data_frame.index.names
                or y_variable not in model_data_frame.index.names
            ):
                print(
                    f"Warning: Cannot create 2D plot for {hyperparameter_config.name} with x={x_variable} and y={y_variable} because one of the variables is not in the evaluation results."
                )
                continue
            for param_name, param_value in plot_config.params.items():
                if param_name not in model_data_frame.index.names:
                    print(
                        f"Warning: Cannot filter for {param_name}={param_value} because {param_name} is not in the evaluation results."
                    )
                    continue
                model_data_frame = model_data_frame.xs(param_value, level=param_name)
            fig, ax = plt.subplots(figsize=(8, 6))
            grid = model_data_frame.pivot_table(
                index=y_variable, columns=x_variable, values="assignment_accuracy"
            )
            im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(grid.columns)))
            ax.set_xticklabels(grid.columns)
            ax.set_yticks(np.arange(len(grid.index)))
            ax.set_yticklabels(grid.index)
            ampl.set_xlabel(label=plot_config.x_label, ax=ax)
            ampl.set_ylabel(label=plot_config.y_label, ax=ax)
            ampl.draw_atlas_label(
                x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress"
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Assignment Accuracy")
            fig.savefig(
                os.path.join(
                    args.output_dir,
                    f"{hyperparameter_config.name}_assignment_accuracy_{plot_config.x_variable}_vs_{plot_config.y_variable}.pdf",
                )
            )
            plt.close(fig)
