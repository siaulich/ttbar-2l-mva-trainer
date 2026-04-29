"""
ROOT file preprocessing module.

This module provides functionality previously implemented in C++ for preprocessing
ROOT files containing particle physics event data. It handles:
- Event pre-selection
- Lepton and jet ordering
- Derived feature computation (invariant masses, delta R)
- Truth information extraction
- Optional NuFlow results and initial parton information
- Saving preprocessed data to ROOT or NPZ formats
"""

import numpy as np
import uproot
import awkward as ak
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import os
from tqdm import tqdm
from ..utils import (
    lorentz_vector_array_from_pt_eta_phi_e,
    compute_mass_from_lorentz_vector_array,
)
from ..evaluation.physics_calculations import c_han, c_hel

from ..configs import PreprocessorConfig, LoadConfig

@dataclass
class InferenceDataConfig:
    """Configuration for data sample preprocessing."""

    preprocessor_config: PreprocessorConfig
    input_path: str
    num_events: Optional[int] = None


class RootInferencePreprocessor:
    """
    Python implementation of ROOT file preprocessing.

    Performs event selection, particle ordering, and feature computation
    on particle physics event data.
    """

    def __init__(self, config: PreprocessorConfig):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.processed_data = {}
        self.mask = None
        self.n_events_processed = 0
        self.n_events_passed = 0


    def extract_event_data(self, events, feature):
        scale_factor = 1.0
        if feature in self.config.scale_by:
            scale_factor = self.config.scale_by[feature]
        return self.extract_event_data(events,feature) * scale_factor


    def process(self, input_path: str):
        """Main processing method."""
        if self.config.verbose:
            print(f"Processing ROOT file: {input_path}")

        # Load data
        with uproot.open(input_path) as file:
            tree = file[self.config.tree_name]
            events = tree.arrays(library="ak")

        self.n_events_processed = len(events)
        self.mask = self._preselection(events)

        if self.config.verbose:
            print(f"Total events in file: {self.n_events_processed}")
        self.n_events_passed = np.sum(self.mask)
        if self.config.verbose:
            print(f"Events passing pre-selection: {self.n_events_passed}")

        # Process events
        self.processed_data = self._process_events(events)

    def _preselection(self, events: ak.Array) -> np.ndarray:
        """
        Apply event pre-selection cuts.

        Args:
            events: Awkward array of events

        Returns:
            Boolean mask of events passing selection
        """
        # Count leptons
        n_electrons = ak.num(
            self.extract_event_data(events,self.config.root_ntuple_config.ElectronConfig.pt)
        )
        n_muons = ak.num(
            self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.pt)
        )
        n_leptons = n_electrons + n_muons

        # Count jets
        n_jets = ak.num(self.extract_event_data(events,self.config.root_ntuple_config.JetConfig.pt))

        # Basic multiplicity cuts
        mask = n_leptons == self.config.n_leptons_required
        mask = mask & (n_jets >= self.config.n_jets_min)

        mask = mask

        return ak.to_numpy(mask)

    def _process_events(self, events: ak.Array) -> Dict[str, np.ndarray]:
        """
        Process events and extract features.

        Args:
            events: Awkward array of events passing pre-selection

        Returns:
            Dictionary of processed features
        """
        processed = {}

        # Process leptons
        leptons = self._process_leptons(events)
        processed.update(leptons)

        # Process jets
        jets = self._process_jets(events)
        processed.update(jets)

        # Process MET
        met = self._process_met(events)
        processed.update(met)

        # Compute derived features
        derived = self._compute_derived_features(events, leptons, jets)
        processed.update(derived)

        # Compute reconstructed mllbb
        reco_mllbb = self._compute_reco_mllbb(leptons, jets)
        processed.update(reco_mllbb)

        if self.config.root_ntuple_config.mc_event_number is not None:
            event_number = ak.to_numpy(
                self.extract_event_data(events,self.config.root_ntuple_config.mc_event_number)
            )
            processed.update({"mc_event_number": event_number})

        return processed

    def _process_leptons(self, events: ak.Array) -> Dict[str, np.ndarray]:
        """
        Process and order leptons.

        Leptons are sorted by charge (positive first).

        Args:
            events: Event array

        Returns:
            Dictionary of lepton features
        """
        # Combine electrons and muons
        lep_pt = ak.concatenate(
            [
                self.extract_event_data(events,self.config.root_ntuple_config.ElectronConfig.pt),
                self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.pt),
            ],
            axis=1,
        )
        lep_eta = ak.concatenate(
            [
                self.extract_event_data(events,self.config.root_ntuple_config.ElectronConfig.eta),
                self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.eta),
            ],
            axis=1,
        )
        lep_phi = ak.concatenate(
            [
                self.extract_event_data(events,self.config.root_ntuple_config.ElectronConfig.phi),
                self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.phi),
            ],
            axis=1,
        )
        lep_e = ak.concatenate(
            [
                self.extract_event_data(events,
                    self.config.root_ntuple_config.ElectronConfig.energy
                ),
                self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.energy),
            ],
            axis=1,
        )
        lep_charge = ak.concatenate(
            [
                self.extract_event_data(events,
                    self.config.root_ntuple_config.ElectronConfig.charge
                ),
                self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.charge),
            ],
            axis=1,
        )
        lep_pid = ak.concatenate(
            [
                ak.ones_like(
                    self.extract_event_data(events,
                        self.config.root_ntuple_config.ElectronConfig.charge
                    )
                )
                * 11,
                ak.ones_like(
                    self.extract_event_data(events,self.config.root_ntuple_config.MuonConfig.charge)
                )
                * 13,
            ],
            axis=1,
        )

        # Sort by charge (positive first) - argsort in descending order
        # sort_idx = ak.argsort(lep_charge, ascending=False)
        # lep_pt = lep_pt[sort_idx]
        # lep_eta = lep_eta[sort_idx]
        # lep_phi = lep_phi[sort_idx]
        # lep_e = lep_e[sort_idx]
        # lep_charge = lep_charge[sort_idx]
        # lep_pid = lep_pid[sort_idx]

        # Pad to 2 leptons and convert to numpy
        max_leptons = 2
        lep_pt_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_pt, max_leptons, clip=True), self.config.padding_value
            )
        )
        lep_eta_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_eta, max_leptons, clip=True), self.config.padding_value
            )
        )
        lep_phi_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_phi, max_leptons, clip=True), self.config.padding_value
            )
        )
        lep_e_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_e, max_leptons, clip=True), self.config.padding_value
            )
        )
        lep_charge_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_charge, max_leptons, clip=True),
                self.config.padding_value,
            )
        )
        lep_pid_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(lep_pid, max_leptons, clip=True), self.config.padding_value
            )
        )

        return {
            "lep_pt": lep_pt_np,
            "lep_eta": lep_eta_np,
            "lep_phi": lep_phi_np,
            "lep_e": lep_e_np,
            "lep_charge": lep_charge_np,
            "lep_pid": lep_pid_np,
        }

    def _process_jets(self, events: ak.Array) -> Dict[str, np.ndarray]:
        n_events = len(events)
        jet_pt = self.extract_event_data(events,self.config.root_ntuple_config.JetConfig.pt)
        jet_eta = self.extract_event_data(events,self.config.root_ntuple_config.JetConfig.eta)
        jet_phi = self.extract_event_data(events,self.config.root_ntuple_config.JetConfig.phi)
        jet_e = self.extract_event_data(events,self.config.root_ntuple_config.JetConfig.energy)

        n_jets = ak.num(jet_pt)
        max_jets = self.config.max_saved_jets

        jet_pt_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_pt, max_jets, clip=True), self.config.padding_value
            )
        )
        jet_eta_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_eta, max_jets, clip=True), self.config.padding_value
            )
        )
        jet_phi_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_phi, max_jets, clip=True), self.config.padding_value
            )
        )
        jet_e_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_e, max_jets, clip=True), self.config.padding_value
            )
        )
        if self.config.root_ntuple_config.JetConfig.btag is not None:
            if isinstance(self.config.root_ntuple_config.JetConfig.btag, str):
                jet_btag = self.extract_event_data(events,
                    self.config.root_ntuple_config.JetConfig.btag
                )
            elif isinstance(self.config.root_ntuple_config.JetConfig.btag, list):
                btag_arrays_np = []
                for btag_branch in self.config.root_ntuple_config.JetConfig.btag:
                    btag_array = self.extract_event_data(events,btag_branch)
                    btag_array_padded = ak.fill_none(
                        ak.pad_none(btag_array, max_jets, clip=True),
                        self.config.padding_value,
                    )
                    btag_arrays_np.append(ak.to_numpy(btag_array_padded))
                jet_btag_np = np.stack(btag_arrays_np, axis=-1)
                jet_btag_np = np.sum(
                    jet_btag_np, axis=-1
                )  # Simple sum of multiple b-tag scores
            else:
                raise ValueError("Invalid btag configuration")

        n_jets = ak.to_numpy(n_jets).astype(np.int32)

        n_bjets = np.sum(
            jet_btag_np >= self.config.root_ntuple_config.JetConfig.btag_threshold,
            axis=-1,
        )

        return {
            "jet_pt": jet_pt_np,
            "jet_eta": jet_eta_np,
            "jet_phi": jet_phi_np,
            "jet_e": jet_e_np,
            "jet_b_tag": jet_btag_np,
            "N_jets": n_jets,
            "N_bjets": n_bjets,
        }

    def _process_met(self, events: ak.Array) -> Dict[str, np.ndarray]:
        met_met = ak.to_numpy(
            self.extract_event_data(events,self.config.root_ntuple_config.METConfig.met_met)
        )
        met_phi = ak.to_numpy(
            self.extract_event_data(events,self.config.root_ntuple_config.METConfig.met_phi)
        )

        return {
            "met_met": met_met,
            "met_phi": met_phi,
        }

    def _compute_derived_features(
        self,
        events: ak.Array,
        leptons: Dict[str, np.ndarray],
        jets: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute derived features (invariant masses, delta R, etc.).

        Args:
            events: Event array
            leptons: Lepton features
            jets: Jet features

        Returns:
            Dictionary of derived features
        """
        n_events = len(events)
        max_jets = jets["jet_pt"].shape[1]

        # Lepton 4-vectors (vectorized)
        l1_pt = leptons["lep_pt"][:, 0:1]  # Shape (n_events, 1)
        l1_eta = leptons["lep_eta"][:, 0:1]
        l1_phi = leptons["lep_phi"][:, 0:1]
        l1_e = leptons["lep_e"][:, 0:1]

        l2_pt = leptons["lep_pt"][:, 1:2]
        l2_eta = leptons["lep_eta"][:, 1:2]
        l2_phi = leptons["lep_phi"][:, 1:2]
        l2_e = leptons["lep_e"][:, 1:2]

        # Convert to px, py, pz with overflow protection
        with np.errstate(over="ignore", invalid="ignore"):
            l1_px = l1_pt * np.cos(l1_phi)
            l1_py = l1_pt * np.sin(l1_phi)
            l1_pz = np.where(
                np.abs(l1_eta) < 10, l1_pt * np.sinh(l1_eta), np.sign(l1_eta) * 1e10
            )

            l2_px = l2_pt * np.cos(l2_phi)
            l2_py = l2_pt * np.sin(l2_phi)
            l2_pz = np.where(
                np.abs(l2_eta) < 10, l2_pt * np.sinh(l2_eta), np.sign(l2_eta) * 1e10
            )

        # Jet 4-vectors
        j_pt = jets["jet_pt"]  # Shape (n_events, max_jets)
        j_eta = jets["jet_eta"]
        j_phi = jets["jet_phi"]
        j_e = jets["jet_e"]

        # Convert to px, py, pz with overflow protection
        with np.errstate(over="ignore", invalid="ignore"):
            j_px = j_pt * np.cos(j_phi)
            j_py = j_pt * np.sin(j_phi)
            j_pz = np.where(
                np.abs(j_eta) < 10, j_pt * np.sinh(j_eta), np.sign(j_eta) * 1e10
            )

        # Compute invariant masses (vectorized)
        # l1 + jet
        with np.errstate(invalid="ignore"):
            l1j_e = l1_e + j_e
            l1j_px = l1_px + j_px
            l1j_py = l1_py + j_py
            l1j_pz = l1_pz + j_pz
            m_l1j_squared = l1j_e**2 - l1j_px**2 - l1j_py**2 - l1j_pz**2
            m_l1j = np.where(
                m_l1j_squared > 0,
                np.sqrt(np.abs(m_l1j_squared)),
                self.config.padding_value,
            )

            # l2 + jet
            l2j_e = l2_e + j_e
            l2j_px = l2_px + j_px
            l2j_py = l2_py + j_py
            l2j_pz = l2_pz + j_pz
            m_l2j_squared = l2j_e**2 - l2j_px**2 - l2j_py**2 - l2j_pz**2
            m_l2j = np.where(
                m_l2j_squared > 0,
                np.sqrt(np.abs(m_l2j_squared)),
                self.config.padding_value,
            )

        # Mark invalid jets
        valid_mask = j_pt != self.config.padding_value
        m_l1j = np.where(valid_mask, m_l1j, self.config.padding_value)
        m_l2j = np.where(valid_mask, m_l2j, self.config.padding_value)

        # Delta R (vectorized)
        dR_l1j = self._delta_r(l1_eta, l1_phi, j_eta, j_phi)
        dR_l2j = self._delta_r(l2_eta, l2_phi, j_eta, j_phi)
        dR_l1j = np.where(valid_mask, dR_l1j, self.config.padding_value)
        dR_l2j = np.where(valid_mask, dR_l2j, self.config.padding_value)

        # Delta R between leptons
        dR_l1l2 = self._delta_r(
            leptons["lep_eta"][:, 0],
            leptons["lep_phi"][:, 0],
            leptons["lep_eta"][:, 1],
            leptons["lep_phi"][:, 1],
        )

        return {
            "m_l1j": m_l1j,
            "m_l2j": m_l2j,
            "dR_l1j": dR_l1j,
            "dR_l2j": dR_l2j,
            "dR_l1l2": dR_l1l2,
        }

    def _compute_reco_mllbb(self, leptons, jets):
        # --- extract lepton 4-vectors ---
        l1_pt = leptons["lep_pt"][:, 0]
        l1_eta = leptons["lep_eta"][:, 0]
        l1_phi = leptons["lep_phi"][:, 0]
        l1_e = leptons["lep_e"][:, 0]

        l2_pt = leptons["lep_pt"][:, 1]
        l2_eta = leptons["lep_eta"][:, 1]
        l2_phi = leptons["lep_phi"][:, 1]
        l2_e = leptons["lep_e"][:, 1]

        # --- select 2 b-jets or fallback leading jets ---
        btag = jets["jet_b_tag"]
        jet_pt = jets["jet_pt"]

        # Score = large bonus if b-tagged + pT
        b_tag_mask = (btag > 2).astype(np.float32)

        # sort descending
        bjet_indices = np.lexsort((-jet_pt, -b_tag_mask), axis=1)[:, :2]

        # fallback to leading jets if less than 2 b-tagged jets

        rows = np.arange(jet_pt.shape[0])
        b1_idx, b2_idx = bjet_indices[:, 0], bjet_indices[:, 1]

        # --- extract jet 4-vectors ---
        b1_pt = jets["jet_pt"][rows, b1_idx]
        b1_eta = jets["jet_eta"][rows, b1_idx]
        b1_phi = jets["jet_phi"][rows, b1_idx]
        b1_e = jets["jet_e"][rows, b1_idx]

        b2_pt = jets["jet_pt"][rows, b2_idx]
        b2_eta = jets["jet_eta"][rows, b2_idx]
        b2_phi = jets["jet_phi"][rows, b2_idx]
        b2_e = jets["jet_e"][rows, b2_idx]

        # --- compute invariant mass ---
        b1_4 = lorentz_vector_array_from_pt_eta_phi_e(b1_pt, b1_eta, b1_phi, b1_e)
        b2_4 = lorentz_vector_array_from_pt_eta_phi_e(b2_pt, b2_eta, b2_phi, b2_e)
        l1_4 = lorentz_vector_array_from_pt_eta_phi_e(l1_pt, l1_eta, l1_phi, l1_e)
        l2_4 = lorentz_vector_array_from_pt_eta_phi_e(l2_pt, l2_eta, l2_phi, l2_e)

        total = b1_4 + b2_4 + l1_4 + l2_4
        return {"reco_mllbb": compute_mass_from_lorentz_vector_array(total)}

    @staticmethod
    def _delta_r(eta1, phi1, eta2, phi2):
        """Compute delta R between two particles."""
        deta = eta1 - eta2
        dphi = phi1 - phi2
        # Wrap phi to [-pi, pi]
        dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
        return np.sqrt(deta**2 + dphi**2)

    def get_processed_data(self) -> Dict[str, np.ndarray]:
        """
        Get the processed data dictionary.

        Returns:
            Dictionary of processed features
        """
        return self.processed_data

    def get_num_events(self) -> int:
        """Get the number of processed events"""
        if self.processed_data:
            first_key = next(iter(self.processed_data))
            return len(self.processed_data[first_key])
        return 0

    def get_inference_data(
        self, load_config: LoadConfig
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Get the data dictionary formatted for inference.

        Args:
            load_config: LoadConfig specifying which features to include

        Returns:
            Dictionary of features for inference
        """
        inference_data = {}
        inference_data["jet_inputs"] = self._load_feature_array(
            self.processed_data,
            load_config.jet_inputs,
            max_objects=load_config.max_jets,
        )
        inference_data["lepton_inputs"] = self._load_feature_array(
            self.processed_data,
            load_config.lepton_inputs,
            max_objects=load_config.NUM_LEPTONS,
        )
        inference_data["met_inputs"] = self._load_feature_array(
            self.processed_data,
            load_config.met_inputs,
        )
        inference_data["global_event_inputs"] = self._load_feature_array(
            self.processed_data,
            load_config.global_event_inputs,
        )
        inference_data["mc_event_number"] = (
            self._load_feature_array(
                self.processed_data,
                ["mc_event_number"],
            )
            if "mc_event_number" in self.processed_data
            else None
        )
        return inference_data, self.mask

    def _load_feature_array(
        self,
        loaded: Dict,
        feature_keys: List[str],
        target_shape: Optional[Tuple[int, ...]] = None,
        max_objects: Optional[int] = None,
    ) -> np.ndarray:
        """Helper to load and transpose feature arrays from npz.

        Args:
            loaded: Loaded npz file data
            feature_keys: List of keys to extract from npz
            target_shape: Shape to reshape into (if needed)
            max_objects: Maximum number of objects to keep (for jets/leptons)

        Returns:
            Transposed and reshaped feature array
        """
        arrays = [loaded[key] for key in feature_keys]
        result = (
            np.array(arrays).transpose(1, 2, 0)
            if len(arrays[0].shape) > 1
            else np.array(arrays).transpose(1, 0)
        )

        if max_objects is not None and len(result.shape) > 2:
            result = result[:, :max_objects, :]

        if target_shape is not None:
            result = result.reshape(target_shape)

        return result


def get_inference_data(
    preprocessor_config: InferenceDataConfig, load_config: LoadConfig
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Preprocess a ROOT file according to the given configuration.
    Args:
        config: PreprocessorConfig with input file and processing settings
    Returns:
        Dictionary of processed features
    """
    preprocessor = RootInferencePreprocessor(preprocessor_config.preprocessor_config)
    preprocessor.process(preprocessor_config.input_path)
    return preprocessor.get_inference_data(load_config)
