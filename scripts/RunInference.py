import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import uproot

plt.rcParams["font.size"] = 18
import matplotlib as mpl
from typing import Tuple, Dict

mpl.rcParams["figure.constrained_layout.use"] = True
from src.preprocessing import InferenceDataConfig, get_inference_data, RootInferencePreprocessor
from src.configs import (
    load_preprocessing_config,
    load_load_config,
    load_inference_config,
    LoadConfig,
    PreprocessorConfig,
    ROOTNtupleConfig
)
from src.utils import lorentz_vector_from_PtEtaPhiE_array, lorentz_vector_from_neutrino_momenta_array
from src.base_classes import KerasMLWrapper
from src.reconstruction import get_reconstructor

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


def compute_infered_tops(
    inference_data: Dict[str, np.ndarray],
    assignement_pred: np.ndarray,
    neutrino_momenta: np.ndarray,
    mask: np.ndarray,
    padding_value: int = -999,
    max_jets: int = 6,
):
    num_events = mask.shape[0]
    """Compute top quark momenta from the raw data and predictions."""
    top_4vect = np.full((num_events, 4), padding_value, np.float32)
    tbar_4vect = np.full((num_events, 4), padding_value, np.float32)
    selected_jet_indices = np.argmax(assignement_pred, axis=-2)
    top_jets = np.take_along_axis(
        inference_data["jet_inputs"][..., :4],
            selected_jet_indices[:, :, np.newaxis],
            axis=1,
        )
    top_jets_4vect = lorentz_vector_from_PtEtaPhiE_array(top_jets)
    lepton_4vect = lorentz_vector_from_PtEtaPhiE_array(inference_data["lep_inputs"][..., :4])
    neutrino_4vect = lorentz_vector_from_neutrino_momenta_array(neutrino_momenta)
    top_4vect[mask] = (top_jets_4vect + lepton_4vect + neutrino_4vect)[:, 0, :]
    tbar_4vect[mask] = (top_jets_4vect + lepton_4vect + neutrino_4vect)[:, 1, :]
    return top_4vect, tbar_4vect

def compute_jet_idx(
    assignment_pred: np.ndarray,
    mask: np.ndarray,
):
    """Compute the jet indices for each event based on the assignment predictions."""
    masked_bjet_idx = np.full((assignment_pred.shape[0], 6), -1, dtype=int)
    selected_jet_indices = np.argmax(assignment_pred, axis=1)
    masked_bjet_idx[..., [0, 3]] = selected_jet_indices[:, :2]
    bjet_idx = np.full((mask.shape[0], 6), -1, dtype=int)
    bjet_idx[mask] = masked_bjet_idx
    return bjet_idx

def compute_neutrino_momenta(
    neutrino_regression: np.ndarray,
    mask: np.ndarray,
):
    """Compute the neutrino momenta for each event based on the regression predictions."""
    neutrino_momenta = np.full((mask.shape[0], 6), np.nan, dtype=np.float32)
    neutrino_momenta[mask] = neutrino_regression.reshape(-1, 6)
    return neutrino_momenta


if __name__ == "__main__":
    args = parse_args()

    preprocessor_config = load_preprocessing_config(args.preprocessor_config)
    load_config = load_load_config(args.load_config)
    inference_config = load_inference_config(args.inference_config)

    data_config = load_config.to_data_config()
    preprocessor = RootInferencePreprocessor(preprocessor_config)
    preprocessor.process(args.input_file)
    inference_data, mask = preprocessor.get_inference_data(load_config)
    
    num_events = inference_data["mc_event_number"].shape[0]
    masked_inference_data = {key: value[mask] for key, value in inference_data.items()}
    reconstructors = {}
    for reconstructor_cfg in inference_config.reconstructors:
        reconstructor_cfg.options["config"] = data_config
        reconstructor = get_reconstructor(reconstructor_cfg.type)(
            **reconstructor_cfg.options
        )
        reconstructors[reconstructor_cfg.name] = reconstructor
    reco_results = {}
    for reconstructor_name in reconstructors:
        print(f"Running inference with reconstructor: {reconstructor_name}")
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
                neutrino_pred = reconstructor.reconstruct_neutrinos(
                    masked_inference_data
                )
            else:
                print("WARNING: Reconstructor does not support neutrino regression.")
                neutrino_pred = None
            reco_results[reconstructor_name] = {
                "assignment": assignment_pred,
                "regression": neutrino_pred,
            }
    truth_idx_results = {}
    # Compute event_***_truth_idx for each reconstructor
    for reconstructor_name, results in reco_results.items():
        assignment_pred = results["assignment"]
        neutrino_regression = results["regression"]
        top_4vect, tbar_4vect = compute_infered_tops(
            inference_data=masked_inference_data,
            assignement_pred=assignment_pred,
            neutrino_momenta=neutrino_regression,
            mask=mask,
            padding_value=-999,
            max_jets=data_config.max_jets,
        )
        top_pt = np.sqrt(top_4vect[:, 0]**2 + top_4vect[:, 1]**2)
        tbar_pt = np.sqrt(tbar_4vect[:, 0]**2 + tbar_4vect[:, 1]**2)
        top_eta = 0.5 * np.log((top_4vect[:, 3] + top_4vect[:, 2]) / (top_4vect[:, 3] - top_4vect[:, 2] + 1e-8))
        tbar_eta = 0.5 * np.log((tbar_4vect[:, 3] + tbar_4vect[:, 2]) / (tbar_4vect[:, 3] - tbar_4vect[:, 2] + 1e-8))
        top_phi = np.arctan2(top_4vect[:, 1], top_4vect[:, 0])
        tbar_phi = np.arctan2(tbar_4vect[:, 1], tbar_4vect[:, 0])
        top_E = top_4vect[:, 3]
        tbar_E = tbar_4vect[:, 3]
        truth_idx_results[reconstructor_name] = {
            "top_pt": top_pt,
            "tbar_pt": tbar_pt,
            "top_eta": top_eta,
            "tbar_eta": tbar_eta,
            "top_phi": top_phi,
            "tbar_phi": tbar_phi,
            "top_e": top_E,
            "tbar_e": tbar_E,
            "event_reco_jet_idx": compute_jet_idx(assignment_pred, mask),
            "nu_3vect": compute_neutrino_momenta(neutrino_regression, mask) * 1e-3, # Convert from MeV to GeV
        }
    # Save results

    output_file = os.path.join(args.output_dir, os.path.basename(args.input_file))
    tree_name = preprocessor.config.tree_name  # adjust if needed

    with uproot.open(args.input_file) as in_file:
        all_keys = in_file.keys(cycle=False)

        with uproot.recreate(output_file) as out_file:
            for key in all_keys:
                obj = in_file[key]

                # Target tree: add new branches
                if key == tree_name and isinstance(obj, uproot.TTree):
                    arrays = obj.arrays(library="np")

                    for reconstructor_name, results in truth_idx_results.items():
                        # Sanitize reconstructor name for use in branch names
                        rname = reconstructor_name.replace("-", "_").replace(" ", "_")
                        for var_name, var_array in results.items():
                            branch_name = f"reco_{rname}_{var_name}"
                            arrays[branch_name] = var_array

                    out_file[key] = arrays

                # Other trees: copy as-is
                elif isinstance(obj, uproot.TTree):
                    out_file[key] = obj.arrays(library="np")

                # Histograms and other objects
                else:
                    out_file[key] = obj

    print(f"Output written to: {output_file}")