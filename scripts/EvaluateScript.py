import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 18
import matplotlib as mpl

mpl.rcParams["figure.constrained_layout.use"] = True
from src.preprocessing.training_data_loader import TrainingDataLoader
from src import evaluation, reconstruction, base_classes
from src.configs import (
    get_load_config_from_yaml,
    load_evaluation_config,
)


def parse_args():
    # """Parse command line arguments for running the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on specified datasets"
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
        default="evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Whether to only evaluate and plot accuracy-related metrics (default: False)",
    )
    parser.add_argument(
        "--ml_metrics",
        action="store_true",
        help="Whether to evaluate and plot ML-specific metrics for ML-based reconstructors (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configurations
    load_config = get_load_config_from_yaml(args.load_config)
    evaluation_config = load_evaluation_config(args.evaluation_config)

    # Load data
    data_processor = TrainingDataLoader(load_config)
    data_config = data_processor.load_from_npz(
        npz_path=load_config.data_path,
        num_events=args.num_events,
        event_numbers=evaluation_config.evaluation_event_numbers,
    )
    X, y = data_processor.get_data()
    num_events = data_processor.get_num_events()
    del data_processor

    print(f"Successfully loaded data for evaluation. Number of events: {num_events}")
    # Initialize reconstructors based on the evaluation configuration
    reconstructors = []
    ml_reconstructors = []
    for reconstructor_cfg in evaluation_config.reconstructors:
        reconstructor_cfg.options["config"] = data_config
        reconstructor = reconstruction.get_reconstructor(reconstructor_cfg.type)(
            **reconstructor_cfg.options
        )
        reconstructors.append(reconstructor)
        print(f"Initialized reconstructor: {reconstructor_cfg.type}")
        if isinstance(reconstructor, base_classes.KerasMLWrapper):
            ml_reconstructors.append(reconstructor)

    output_dir = args.output_dir or "./evaluation_results"
    if ml_reconstructors and args.ml_metrics:
        ml_evaluator = evaluation.MLEvaluator(ml_reconstructors, X, y)
        ml_metrics_output_dir = os.path.join(
            args.output_dir or "./evaluation_results", "ml_metrics"
        )
        os.makedirs(ml_metrics_output_dir, exist_ok=True)
        fig, ax = ml_evaluator.plot_training_history()
        fig.savefig(os.path.join(ml_metrics_output_dir, "training_history.pdf"))
        print(f"Saved training history plot")

        fig, ax = ml_evaluator.plot_model_parameters_comparison()
        fig.savefig(
            os.path.join(ml_metrics_output_dir, "model_parameters_comparison.pdf")
        )
        print(f"Saved model parameters comparison plot")

        fig, ax = ml_evaluator.plot_inference_time_comparison()
        fig.savefig(
            os.path.join(ml_metrics_output_dir, "inference_time_comparison.pdf")
        )
        print(f"Saved inference time comparison plot")

        del ml_evaluator
    del ml_reconstructors

    prediction_manager = evaluation.PredictionManager(reconstructors, X, y)
    print("Initialized PredictionManager")

    evaluator = evaluation.ReconstructionPlotter(prediction_manager)
    os.makedirs(output_dir, exist_ok=True)

    deviation_directory = os.path.join(output_dir, "deviations")
    os.makedirs(deviation_directory, exist_ok=True)
    confusion_matrix_directory = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(confusion_matrix_directory, exist_ok=True)
    accuracy_directory = os.path.join(output_dir, "accuracy")
    os.makedirs(accuracy_directory, exist_ok=True)
    neutrino_deviation_directory = os.path.join(output_dir, "neutrino_deviations")
    os.makedirs(neutrino_deviation_directory, exist_ok=True)
    distributions_directory = os.path.join(output_dir, "distributions")
    os.makedirs(distributions_directory, exist_ok=True)
    binned_performance_directory = os.path.join(output_dir, "binned_performance")
    os.makedirs(binned_performance_directory, exist_ok=True)
    evaluator.plot_accuracy_evaluation(save_dir=accuracy_directory)
    print(f"Saved all accuracy evaluation plots")
    if not args.accuracy:
        evaluator.plot_all_distributions(save_dir=distributions_directory)
        print(f"Saved all distribution plots")
        evaluator.plot_all_deviations(save_dir=deviation_directory)
        print(f"Saved all deviation evaluation plots")
        evaluator.plot_neutrino_deviation_evaluation(
            save_dir=neutrino_deviation_directory
        )
        print(f"Saved all neutrino deviation evaluation plots")
        evaluator.plot_all_confusion_matrices(save_dir=confusion_matrix_directory)
        print(f"Saved all confusion matrix plots")

    for idx, binning_cfg in enumerate(evaluation_config.binning_variables):
        binned_variable_output_dir = os.path.join(
            binned_performance_directory, f"{binning_cfg.feature_name}"
        )
        os.makedirs(binned_variable_output_dir, exist_ok=True)
        evaluator.plot_binned_performance_evaluation(
            **binning_cfg.__dict__,
            save_dir=binned_variable_output_dir,
            accuracy_only=args.accuracy,
        )
        print(
            f"Saved binned evaluation plots for {binning_cfg.feature_name} [{idx + 1}/{len(evaluation_config.binning_variables)}]"
        )

    if evaluation_config.binned_2d_binning_variables:
        binned_2d_performance_directory = os.path.join(
            output_dir, "binned_2d_performance"
        )
        os.makedirs(binned_2d_performance_directory, exist_ok=True)

    for idx, (binning_cfgs) in enumerate(evaluation_config.binned_2d_binning_variables):
        if len(binning_cfgs) != 2:
            print(
                f"Warning: Skipping invalid binned 2D variable configuration at index {idx} - expected exactly 2 binning variables, got {len(binning_cfgs)}"
            )
            continue
        binning_cfg1, binning_cfg2 = binning_cfgs[0], binning_cfgs[1]
        binned_2d_variable_output_dir = os.path.join(
            binned_2d_performance_directory,
            f"{binning_cfg1.feature_name}_vs_{binning_cfg2.feature_name}",
        )
        os.makedirs(binned_2d_variable_output_dir, exist_ok=True)
        evaluator.plot_2d_binned_performance_evaluation(
            binning_cfg1,
            binning_cfg2,
            save_dir=binned_2d_variable_output_dir,
        )
        print(
            f"Saved binned 2D evaluation plots for {binning_cfg1.feature_name} vs. {binning_cfg2.feature_name} [{idx + 1}/{len(evaluation_config.binned_2d_binning_variables)}]"
        )

    print(f"Saved all evaluation plots to {output_dir}")
