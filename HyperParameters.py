import sys
import argparse
import os

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
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with different hyperparameters on specified datasets"
    )
    parser.add_argument("--load_config", type=str, required=True)
    parser.add_argument("--evaluation_config", type=str, required=True)
    parser.add_argument("--num_events", type=int, default=2_000_000)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/hyperparameter_evaluation_results",
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--k_folds",
        type=int,
        default=None,
        help="Number of k-folds used during training. If set, evaluation averages "
        "metrics across folds and computes std as uncertainty. "
        "Expects fold directories named <dir_name>_fold<i>/",
    )
    return parser.parse_args()


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricConfig:
    key: str  # column name in the dataframe
    label: str  # pretty axis / colorbar label
    scale: float = 1.0  # multiply values by this before plotting
    log_scale: bool = False  # use log scale on this axis


METRICS_CONFIG: dict[str, MetricConfig] = {
    m.key: m
    for m in [
        MetricConfig(
            key="assignment_accuracy",
            label="Assignment Accuracy",
        ),
        MetricConfig(
            key="regression_mse",
            label="Regression MSE",
        ),
        MetricConfig(
            key="num_trainable_parameters",
            label="Number of Trainable Parameters [Mio]",
            scale=1e-6,
            log_scale=True,
        ),
        MetricConfig(
            key="inference_time_per_event",
            label="Inference Time per Event [ms]",
            scale=1e3,
        ),
        MetricConfig(
            key="training_epochs",
            label="Number of Training Epochs",
        ),
    ]
}


@dataclass
class SummaryPlotConfig:
    x_metric: str  # key in METRICS_CONFIG
    y_metric: str  # key in METRICS_CONFIG
    filename: str
    legend_loc: str = "lower right"


SUMMARY_PLOTS: list[SummaryPlotConfig] = [
    SummaryPlotConfig(
        x_metric="num_trainable_parameters",
        y_metric="assignment_accuracy",
        filename="assignment_accuracy_vs_num_parameters.pdf",
        legend_loc="lower right",
    ),
    SummaryPlotConfig(
        x_metric="num_trainable_parameters",
        y_metric="training_epochs",
        filename="training_epochs_vs_num_parameters.pdf",
    ),
    SummaryPlotConfig(
        x_metric="num_trainable_parameters",
        y_metric="inference_time_per_event",
        filename="inference_time_vs_num_parameters.pdf",
    ),
    SummaryPlotConfig(
        x_metric="training_epochs",
        y_metric="assignment_accuracy",
        filename="assignment_accuracy_vs_training_epochs.pdf",
    ),
]


def evaluate_single_dir(dir_name, model_type, options, data_config, X, y):
    """Evaluate all metrics for a single model directory. Returns a dict of metric -> value."""
    if not os.path.exists(dir_name):
        return None

    for file in os.listdir(dir_name):
        if not file.endswith(".keras"):
            continue
        file_name = os.path.join(dir_name, file)
        model = model_type(data_config, **options)
        model.load_model(file_name)

        ml_evaluator = evaluation.MLEvaluator([model], X, y)
        results = {
            "assignment_accuracy": ml_evaluator.evaluate_idx(0)["assignment_accuracy"],
            "regression_mse": ml_evaluator.evaluate_idx(0)["regression_mse"],
            "num_trainable_parameters": ml_evaluator.evaluate_num_parameters_idx(0)[
                "num_trainable_parameters"
            ],
            "inference_time_per_event": ml_evaluator.evaluate_inference_time_idx(0)[
                "time_per_sample"
            ],
            "training_epochs": ml_evaluator.evaluate_num_training_epochs_idx(0)[
                "num_training_epochs"
            ],
        }
        del ml_evaluator
        del model
        return results  # One .keras file per directory expected

    return None


def evaluate_combination(
    dir_name, model_type, options, data_config, X, y, k_folds=None
):
    """
    Evaluate a hyperparameter combination, optionally aggregating across k folds.
    Returns a flat dict of metric -> value (and metric_std -> value if k_folds is set).
    """
    if k_folds is None:
        results = evaluate_single_dir(dir_name, model_type, options, data_config, X, y)
        return results  # May be None if dir missing

    # Collect results across folds
    fold_results = []
    for fold_idx in range(k_folds):
        fold_dir = f"{dir_name}_fold{fold_idx}"
        fold_result = evaluate_single_dir(
            fold_dir, model_type, options, data_config, X, y
        )
        if fold_result is not None:
            fold_results.append(fold_result)
        else:
            print(f"  Warning: fold directory {fold_dir} missing or empty, skipping.")

    if not fold_results:
        return None

    # Aggregate: mean + std across folds
    aggregated = {}
    for metric in METRICS_CONFIG.keys():
        values = [r[metric] for r in fold_results]
        aggregated[metric] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)

    aggregated["num_folds_evaluated"] = len(fold_results)
    return aggregated


if __name__ == "__main__":
    args = parse_args()

    load_config = get_load_config_from_yaml(args.load_config)
    evaluation_config = load_hyperparameter_evaluation_config(args.evaluation_config)

    os.makedirs(args.output_dir, exist_ok=True)

    models = dict()
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
        model_data_frame = None

        if not args.recompute:
            csv_path = os.path.join(
                args.output_dir,
                f"{hyperparameter_config.name}_evaluation_results.csv",
            )
            if os.path.exists(csv_path):
                model_data_frame = pd.read_csv(csv_path, index_col=index_names)

        if model_data_frame is None:
            model_data_frame = pd.DataFrame(index=multi_index)

            # Build column series — include _std columns when k-folding
            series = {
                metric: pd.Series(index=multi_index, dtype=float)
                for metric in METRICS_CONFIG.keys()
            }
            # num_trainable_parameters and training_epochs are integers but keep as float for NaN support
            if args.k_folds is not None:
                for metric in METRICS_CONFIG.keys():
                    series[f"{metric}_std"] = pd.Series(index=multi_index, dtype=float)
                series["num_folds_evaluated"] = pd.Series(
                    index=multi_index, dtype=float
                )

            for hyperparameter_combination in tqdm.tqdm(
                multi_index, desc=f"Evaluating {hyperparameter_config.name}"
            ):
                dir_name = hyperparameter_config.dir_name_pattern.format(
                    *hyperparameter_combination
                )
                options = deepcopy(hyperparameter_config.options)
                results = evaluate_combination(
                    dir_name,
                    model_type,
                    options,
                    data_config,
                    X,
                    y,
                    k_folds=args.k_folds,
                )

                if results is None:
                    print(f"No results for {dir_name}, skipping.")
                    continue

                for key, value in results.items():
                    if key in series:
                        series[key].loc[hyperparameter_combination] = value

            for key, s in series.items():
                model_data_frame[key] = s

            csv_path = os.path.join(
                args.output_dir,
                f"{hyperparameter_config.name}_evaluation_results.csv",
            )
            model_data_frame.to_csv(csv_path)

        models[hyperparameter_config.name] = model_data_frame

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def get_yerr(df, metric):
        """Return yerr array if std column exists, else None."""
        std_col = f"{metric}_std"
        if std_col in df.columns:
            return df[std_col].values
        return None


# Summary scatter plots
for plot_cfg in SUMMARY_PLOTS:
    x_cfg = METRICS_CONFIG[plot_cfg.x_metric]
    y_cfg = METRICS_CONFIG[plot_cfg.y_metric]

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in models.items():
        x = df[x_cfg.key].values * x_cfg.scale
        y = df[y_cfg.key].values * y_cfg.scale
        yerr = get_yerr(df, y_cfg.key)
        if yerr is not None:
            yerr = yerr * y_cfg.scale
        ax.errorbar(x, y, yerr=yerr, fmt="o", label=model_name, alpha=0.7, capsize=3)

    if x_cfg.log_scale:
        ax.set_xscale("log")
    if y_cfg.log_scale:
        ax.set_yscale("log")
    ampl.set_xlabel(label=x_cfg.label, ax=ax)
    ampl.set_ylabel(label=y_cfg.label, ax=ax)
    ampl.draw_atlas_label(x=0.02, y=0.95, ax=ax, status="Simulation Work in Progress")
    ax.legend(loc=plot_cfg.legend_loc)
    fig.savefig(os.path.join(args.output_dir, plot_cfg.filename))
    plt.close(fig)

# 2D heatmap plots
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
                f"Warning: Cannot create 2D plot for {hyperparameter_config.name} "
                f"with x={x_variable} and y={y_variable}."
            )
            continue

        filtered_df = model_data_frame.copy()
        for param_name, param_value in plot_config.params.items():
            if param_name not in filtered_df.index.names:
                print(f"Warning: Cannot filter for {param_name}={param_value}.")
                continue
            filtered_df = filtered_df.xs(param_value, level=param_name)

        plot_columns = [
            col
            for col in filtered_df.columns
            if not col.endswith("_std") and col != "num_folds_evaluated"
        ]
        for variable in plot_columns:
            for col in [variable, f"{variable}_std"]:
                if col not in filtered_df.columns:
                    continue

                # Look up pretty label — fall back to raw column name
                base_key = col.removesuffix("_std")
                metric_cfg = METRICS_CONFIG.get(base_key)
                is_std = col.endswith("_std")
                if metric_cfg:
                    cbar_label = (
                        f"Std. Dev. of {metric_cfg.label}"
                        if is_std
                        else metric_cfg.label
                    )
                    # Apply scale to grid values
                    scale = metric_cfg.scale
                else:
                    cbar_label = col
                    scale = 1.0

                grid = (
                    filtered_df.pivot_table(
                        index=y_variable, columns=x_variable, values=col
                    )
                    * scale
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(
                    grid.values, origin="lower", aspect="auto", cmap="viridis"
                )
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
                cbar.set_label(cbar_label)
                fig.savefig(
                    os.path.join(
                        args.output_dir,
                        f"{hyperparameter_config.name}_{col}_"
                        f"{plot_config.x_variable}_vs_{plot_config.y_variable}.pdf",
                    )
                )
                plt.close(fig)
