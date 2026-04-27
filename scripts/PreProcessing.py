import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.RootPreprocessor import (
    RootPreprocessor,
    preprocess_root_file,
    preprocess_root_directory,
    DataSampleConfig,
)
from dacite import from_dict, Config
from src.configs import ROOTNtupleConfig, load_yaml_config, PreprocessorConfig


def parse_args():
    # """Parse command line arguments for running the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on specified datasets"
    )

    # Configuration file arguments
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the preprocessing run (used for logging and output organization)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file containing all necessary parameters for preprocessing",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=2_000_000,
        help="Number of events to preprocess (default: 2,000,000)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the input ROOT files to preprocess",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessed_data",
        help="Directory to save preprocessed data (default: ./preprocessed_data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    preprocessor_config = load_yaml_config(args.config)
    preprocessor_config = from_dict(
        data_class=PreprocessorConfig,
        data=preprocessor_config["PreprocessorConfig"],
    )

    # Initialize the preprocessor
    data_sample_config = DataSampleConfig(
        name=args.name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_events=args.num_events,
        preprocessor_config=preprocessor_config,
    )

    # Preprocess data
    preprocess_root_directory(data_sample_config)
