import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.size"] = 18
import matplotlib as mpl
from typing import Tuple, Dict

mpl.rcParams["figure.constrained_layout.use"] = True
from src.preprocessing import InferenceDataConfig, get_inference_data
from src.configs import (
    load_preprocessing_config,
    load_load_config,
    load_inference_config,
    LoadConfig,
    PreprocessorConfig,
)
from src.base_classes import KerasMLWrapper
import src.reconstruction as reconstruction

preprocessor_config = load_preprocessing_config("examples/preprocessing.yaml")
preprocessor_config.save_mc_truth = False


def parse_args():
    """Parse command line arguments for running the inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference on ROOT files using a trained model"
    )

    # Configuration file arguments
    parser.add_argument(
        "--preprocessor_config",
        type=str,
        required=True,
        help="Path to the preprocessor configuration YAML file",
    )
    parser.add_argument(
        "--load_config",
        type=str,
        required=True,
        help="Path to the load configuration YAML file",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input ROOT file for inference",
    )
    parser.add_argument(
        "--inference_config",
        type=str,
        required=True,
        help="Path to the inference configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results and logs",
    )

    return parser.parse_args()


def compute_truth_indices(masked_inference_data, assignment_pred, regression_pred) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lep_pid = np.abs(masked_inference_data["lep_pid"])
    lep_charge = masked_inference_data["lep_charge"]
    is_electron = lep_pid == 11
    is_muon = lep_pid == 13
    is_positive = lep_charge > 0
    is_negative = lep_charge < 0
    event_el_truth_idx = np.full((masked_inference_data["lep_pid"].shape[0], 6), -1, dtype=int)
    event_mu_truth_idx = np.full((masked_inference_data["lep_pid"].shape[0], 6), -1, dtype=int)
    event_jet_truth_idx = np.full((masked_inference_data["lep_pid"].shape[0], 6), -1, dtype=int)
    neutrino_momenta = np.full((masked_inference_data["lep_pid"].shape[0], 6), np.nan)
    assignment_index = np.argmax(assignment_pred, axis=1)

    top_electron_mask = is_electron & is_positive
    anti_top_electron_mask = is_electron & is_negative
    top_muon_mask = is_muon & is_positive
    anti_top_muon_mask = is_muon & is_negative

    num_electrons = np.sum(is_electron, axis=1)
    num_muons = np.sum(is_muon, axis=1)

    mumu_mask = (num_muons == 2) & (num_electrons == 0)
    ee_mask = (num_electrons == 2) & (num_muons == 0)
    emu_mask = (num_electrons == 1) & (num_muons == 1)

    event_el_truth_idx[ee_mask, 1] = np.where(top_electron_mask[ee_mask])[1]
    event_el_truth_idx[ee_mask, 4] = np.where(anti_top_electron_mask[ee_mask])[1]

    event_mu_truth_idx[mumu_mask, 1] = np.where(top_muon_mask[mumu_mask])[1]
    event_mu_truth_idx[mumu_mask, 4] = np.where(anti_top_muon_mask[mumu_mask])[1]

    event_el_truth_idx[emu_mask & np.any(top_electron_mask, axis=-1), 1] = 0
    event_el_truth_idx[emu_mask & np.any(anti_top_electron_mask, axis=-1), 4] = 0
    event_mu_truth_idx[emu_mask & np.any(top_muon_mask, axis=-1), 1] = 0
    event_mu_truth_idx[emu_mask & np.any(anti_top_muon_mask, axis=-1), 4] = 0

    pass


if __name__ == "__main__":
    args = parse_args()

    preprocessor_config = load_preprocessing_config(args.preprocessor_config)
    load_config = load_load_config(args.load_config)
    inference_config = load_inference_config(args.inference_config)

    inference_data_config = InferenceDataConfig(
        preprocessor_config=preprocessor_config, input_path=args.input_file
    )
    data_config = load_config.to_data_config()
    inference_data, mask = get_inference_data(inference_data_config, load_config)
    num_events = inference_data["mc_event_number"].shape[0]
    masked_inference_data = {key: value[mask] for key, value in inference_data.items()}
    reconstructors = {}
    for reconstructor_cfg in inference_config.reconstructors:
        reconstructor_cfg.options["config"] = data_config
        reconstructor = reconstruction.get_reconstructor(reconstructor_cfg.type)(
            **reconstructor_cfg.options
        )
        reconstructors[reconstructor_cfg.name] = reconstructor
    reco_results = {}
    for reconstructor_name in reconstructors:
        print(
            f"Running inference with reconstructor: {reconstructor_name}"
        )
        reconstructor = reconstructors[reconstructor_name]
        if isinstance(reconstructor, KerasMLWrapper):
            assignment_pred, neutrino_regression = reconstructor.complete_forward_pass(
                masked_inference_data
            )
            reco_results[reconstructor_name] = {
                    "assignment": assignment_pred,
                    "regression": neutrino_regression,
            }
        else:
            assignment_pred = reconstructor.predict_indices(masked_inference_data)
            if hasattr(reconstructor, "reconstruct_neutrinos"):
                neutrino_pred = reconstructor.reconstruct_neutrinos(masked_inference_data)
            else:
                print("WARNING: Reconstructor does not support neutrino regression.")
                neutrino_pred = None
            reco_results[reconstructor_name] = {
                    "assignment": assignment_pred,
                    "regression": neutrino_pred,
                }
    truth_idx_results = {}
    # Compute event_***_truth_idx for each reconstructor
    for reconstructor_name in reco_results:
        event_jet_truth_idx = np.full((num_events, 6), -1, dtype=int)
        event_el_truth_idx = np.full((num_events, 6), -1, dtype=int)
        event_mu_truth_idx = np.full((num_events, 6), -1, dtype=int)
        neutrino_momenta = np.full((num_events, 6), np.nan)
        assignment_pred = reco_results[reconstructor_name]["assignment"]
        regression_pred = reco_results[reconstructor_name]["regression"]
        event_jet_truth_idx[mask], event_el_truth_idx[mask], event_mu_truth_idx[mask], neutrino_momenta[mask] = compute_truth_indices(
            masked_inference_data, assignment_pred, regression_pred
        )


