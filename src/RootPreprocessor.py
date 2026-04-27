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
from src.utils import (
    lorentz_vector_array_from_pt_eta_phi_e,
    compute_mass_from_lorentz_vector_array,
)
from src.evaluation.physics_calculations import c_han, c_hel

import src.configs as configs


@dataclass
class DataSampleConfig:
    """Configuration for data sample preprocessing."""

    preprocessor_config: configs.PreprocessorConfig
    name: str
    output_dir: str
    input_dir: str
    num_events: Optional[int] = None
    k_fold: Optional[int] = None  # For cross-validation splits


class RootPreprocessor:
    """
    Python implementation of ROOT file preprocessing (previously in C++).

    Performs event selection, particle ordering, and feature computation
    on particle physics event data.
    """

    def __init__(self, config: configs.PreprocessorConfig):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.processed_data = {}
        self.n_events_processed = 0
        self.n_events_passed = 0

    def process(self, input_path: str):
        """Main processing method."""
        if self.config.verbose:
            print(f"Processing ROOT file: {input_path}")

        # Load data
        with uproot.open(input_path) as file:
            tree = file[self.config.tree_name]
            events = tree.arrays(library="ak")

        self.n_events_processed = len(events)

        if self.config.verbose:
            print(f"Total events in file: {self.n_events_processed}")

        # Apply pre-selection
        selection_mask = self._preselection(events)
        events = events[selection_mask]
        self.n_events_passed = len(events)

        if self.config.verbose:
            print(f"Events passing pre-selection: {self.n_events_passed}")

        # Process events
        self.processed_data = self._process_events(events)

        self.clean_event_data()

        return self.processed_data

    def clean_event_data(self):
        valid_mask = np.ones(self.n_events_passed, dtype=bool)
        for key, array in self.processed_data.items():
            valid_mask &= ~np.any(np.isnan(array), axis=tuple(range(1, array.ndim)))
            valid_mask &= ~np.any(np.isinf(array), axis=tuple(range(1, array.ndim)))

        if self.config.verbose:
            n_invalid = self.n_events_passed - np.sum(valid_mask)
            print(f"Events with invalid data (NaN/Inf): {n_invalid}")

        if self.config.data_cuts is not None:
            for feature, (min_val, max_val) in self.config.data_cuts.items():
                if feature in self.processed_data:
                    feature_array = self.processed_data[feature]
                    if min_val is not None:
                        valid_mask &= np.all(
                            feature_array >= min_val,
                            axis=tuple(range(1, feature_array.ndim)),
                        )
                    if max_val is not None:
                        valid_mask &= np.all(
                            feature_array <= max_val,
                            axis=tuple(range(1, feature_array.ndim)),
                        )
                else:
                    if self.config.verbose:
                        print(
                            f"Warning: Feature '{feature}' specified in data_cuts not found in processed data."
                        )

        for key in self.processed_data.keys():
            self.processed_data[key] = self.processed_data[key][valid_mask]

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
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.pt)
        )
        n_muons = ak.num(
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.pt)
        )
        n_leptons = n_electrons + n_muons

        # Count jets
        n_jets = ak.num(events.__getitem__(self.config.root_ntuple_config.JetConfig.pt))

        # Basic multiplicity cuts
        mask = n_leptons == self.config.n_leptons_required
        mask = mask & (n_jets >= self.config.n_jets_min)

        # Check valid truth indices for b-jets
        parton_match = ak.pad_none(
            events.__getitem__(
                self.config.root_ntuple_config.MatchingConfig.jet_parton_match_branch
            ),
            6,
        )
        jet_truth_0 = ak.fill_none(
            parton_match[
                :,
                self.config.root_ntuple_config.MatchingConfig.jet_parton_match_index_positions[
                    0
                ],
            ],
            -1,
        )
        jet_truth_3 = ak.fill_none(
            parton_match[
                :,
                self.config.root_ntuple_config.MatchingConfig.jet_parton_match_index_positions[
                    1
                ],
            ],
            -1,
        )
        mask = mask & (jet_truth_0 != -1) & (jet_truth_3 != -1)
        mask = mask & (jet_truth_0 <= self.config.max_jets_for_truth)
        mask = mask & (jet_truth_3 <= self.config.max_jets_for_truth)
        # Check lepton truth indices
        electron_truth_0 = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.electron_parton_match_branch
                ),
                6,
            )[
                ...,
                self.config.root_ntuple_config.MatchingConfig.electron_parton_match_index_positions[
                    0
                ],
            ],
            -1,
        )
        electron_truth_1 = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.electron_parton_match_branch
                ),
                6,
            )[
                ...,
                self.config.root_ntuple_config.MatchingConfig.electron_parton_match_index_positions[
                    1
                ],
            ],
            -1,
        )
        muon_truth_0 = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.muon_parton_match_branch
                ),
                6,
            )[
                ...,
                self.config.root_ntuple_config.MatchingConfig.muon_parton_match_index_positions[
                    0
                ],
            ],
            -1,
        )
        muon_truth_1 = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.muon_parton_match_branch
                ),
                6,
            )[
                ...,
                self.config.root_ntuple_config.MatchingConfig.muon_parton_match_index_positions[
                    1
                ],
            ],
            -1,
        )

        has_truth_lep_0 = (electron_truth_0 != -1) | (muon_truth_0 != -1)
        has_truth_lep_1 = (electron_truth_1 != -1) | (muon_truth_1 != -1)
        mask = mask & has_truth_lep_0 & has_truth_lep_1
        # Charge requirements
        mask = mask & self._check_charge_requirements(events)

        return ak.to_numpy(mask)

    def _check_charge_requirements(self, events: ak.Array) -> ak.Array:
        """
        Check that leptons have opposite charges and total charge is zero.

        Args:
            events: Awkward array of events

        Returns:
            Boolean mask
        """
        n_electrons = ak.num(
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.charge)
        )
        n_muons = ak.num(
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge)
        )

        # Initialize mask
        mask = ak.ones_like(n_electrons, dtype=bool)

        # ee channel: check that two electrons have opposite charge
        ee_mask = n_electrons == 2
        # Pad arrays to avoid index errors
        el_charge_padded = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.ElectronConfig.charge
                ),
                2,
            ),
            0,
        )
        ee_same_charge = (
            el_charge_padded[..., 0] == el_charge_padded[..., 1]
        ) & ee_mask
        mask = mask & ~ee_same_charge

        # mumu channel: check that two muons have opposite charge
        mumu_mask = n_muons == 2
        mu_charge_padded = ak.fill_none(
            ak.pad_none(
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge), 2
            ),
            0,
        )
        mumu_same_charge = (
            mu_charge_padded[:, 0] == mu_charge_padded[:, 1]
        ) & mumu_mask
        mask = mask & ~mumu_same_charge

        # emu channel: check that electron and muon have opposite charge
        emu_mask = (n_electrons == 1) & (n_muons == 1)
        el_charge_first = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.ElectronConfig.charge
                ),
                1,
            )[..., 0],
            0,
        )
        mu_charge_first = ak.fill_none(
            ak.pad_none(
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge), 1
            )[..., 0],
            0,
        )
        emu_same_charge = (el_charge_first == mu_charge_first) & emu_mask
        mask = mask & ~emu_same_charge

        # Total charge must be zero
        el_charge_sum = ak.sum(
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.charge),
            axis=-1,
        )
        mu_charge_sum = ak.sum(
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge),
            axis=-1,
        )
        total_charge = el_charge_sum + mu_charge_sum
        mask = mask & (total_charge == 0)

        return mask

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

        # Process event weights
        if self.config.root_ntuple_config.event_weight is not None:
            event_weight = ak.to_numpy(
                events.__getitem__(self.config.root_ntuple_config.event_weight)
            )
            processed.update({"weight_mc": event_weight})

        # Compute reconstructed mllbb
        reco_mllbb = self._compute_reco_mllbb(leptons, jets)
        processed.update(reco_mllbb)

        # Extract truth information
        truth = self._extract_truth_info(events)
        processed.update(truth)

        # Extract neutrino reconstruction results if configured
        neutrino_reco = self._extract_neutrino_reco(events)
        if neutrino_reco is not None:
            processed.update(neutrino_reco)

        if self.config.root_ntuple_config.mc_event_number is not None:
            event_number = ak.to_numpy(
                events.__getitem__(self.config.root_ntuple_config.mc_event_number)
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
        # Combine electrons and muons using vectorized operations
        n_events = len(events)

        el_truth_padded = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.electron_parton_match_branch
                ),
                6,
            ),
            -1,
        )
        mu_truth_padded = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.muon_parton_match_branch
                ),
                6,
            ),
            -1,
        )
        el_truth_padded = el_truth_padded[
            :,
            self.config.root_ntuple_config.MatchingConfig.electron_parton_match_index_positions,
        ]
        mu_truth_padded = mu_truth_padded[
            :,
            self.config.root_ntuple_config.MatchingConfig.muon_parton_match_index_positions,
        ]

        # Create lepton arrays with truth matching - use broadcasting that works with jagged arrays
        # For electrons - expand truth indices to match electron array shape
        el_idx = ak.local_index(
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.pt)
        )
        el_truth_0 = ak.broadcast_arrays(
            el_truth_padded[:, 0],
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.pt),
        )[0]
        el_truth_1 = ak.broadcast_arrays(
            el_truth_padded[:, 1],
            events.__getitem__(self.config.root_ntuple_config.ElectronConfig.pt),
        )[0]
        el_truth_idx = ak.where(
            el_idx == el_truth_0, 1, ak.where(el_idx == el_truth_1, -1, -1)
        )

        # For muons - expand truth indices to match muon array shape
        mu_idx = ak.local_index(
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.pt)
        )
        mu_truth_1 = ak.broadcast_arrays(
            mu_truth_padded[:, 1],
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.pt),
        )[0]
        mu_truth_0 = ak.broadcast_arrays(
            mu_truth_padded[:, 0],
            events.__getitem__(self.config.root_ntuple_config.MuonConfig.pt),
        )[0]
        mu_truth_idx = ak.where(
            mu_idx == mu_truth_0, 1, ak.where(mu_idx == mu_truth_1, -1, -1)
        )

        # Combine electrons and muons
        lep_pt = ak.concatenate(
            [
                events.__getitem__(self.config.root_ntuple_config.ElectronConfig.pt),
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.pt),
            ],
            axis=1,
        )
        lep_eta = ak.concatenate(
            [
                events.__getitem__(self.config.root_ntuple_config.ElectronConfig.eta),
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.eta),
            ],
            axis=1,
        )
        lep_phi = ak.concatenate(
            [
                events.__getitem__(self.config.root_ntuple_config.ElectronConfig.phi),
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.phi),
            ],
            axis=1,
        )
        lep_e = ak.concatenate(
            [
                events.__getitem__(
                    self.config.root_ntuple_config.ElectronConfig.energy
                ),
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.energy),
            ],
            axis=1,
        )
        lep_charge = ak.concatenate(
            [
                events.__getitem__(
                    self.config.root_ntuple_config.ElectronConfig.charge
                ),
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge),
            ],
            axis=1,
        )
        lep_pid = ak.concatenate(
            [
                events.__getitem__(self.config.root_ntuple_config.ElectronConfig.charge)
                * 11,
                events.__getitem__(self.config.root_ntuple_config.MuonConfig.charge)
                * 13,
            ],
            axis=1,
        )
        lep_truth = ak.concatenate([el_truth_idx, mu_truth_idx], axis=1)

        # Sort by charge (positive first) - argsort in descending order
        sort_idx = ak.argsort(lep_charge, ascending=False)
        lep_pt = lep_pt[sort_idx]
        lep_eta = lep_eta[sort_idx]
        lep_phi = lep_phi[sort_idx]
        lep_e = lep_e[sort_idx]
        lep_charge = lep_charge[sort_idx]
        lep_pid = lep_pid[sort_idx]
        lep_truth = lep_truth[sort_idx]

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
        lep_truth_np = ak.to_numpy(
            ak.fill_none(ak.pad_none(lep_truth, max_leptons, clip=True), -1)
        )

        # Build truth index array
        event_lepton_truth_idx = np.full((n_events, 2), -1, dtype=np.int32)
        for idx in range(max_leptons):
            mask_top = lep_truth_np[:, idx] == 1
            mask_tbar = lep_truth_np[:, idx] == -1
            event_lepton_truth_idx[mask_top, 0] = idx
            event_lepton_truth_idx[mask_tbar, 1] = idx

        return {
            "lep_pt": lep_pt_np,
            "lep_eta": lep_eta_np,
            "lep_phi": lep_phi_np,
            "lep_e": lep_e_np,
            "lep_charge": lep_charge_np,
            "lep_pid": lep_pid_np,
            "event_lepton_truth_idx": event_lepton_truth_idx,
        }

    def _process_jets(self, events: ak.Array) -> Dict[str, np.ndarray]:
        n_events = len(events)

        jet_truth_padded = ak.fill_none(
            ak.pad_none(
                events.__getitem__(
                    self.config.root_ntuple_config.MatchingConfig.jet_parton_match_branch
                ),
                6,
            ),
            -1,
        )

        jet_idx = ak.local_index(
            events.__getitem__(self.config.root_ntuple_config.JetConfig.pt)
        )
        jet_truth_0 = ak.broadcast_arrays(
            jet_truth_padded[
                :,
                self.config.root_ntuple_config.MatchingConfig.jet_parton_match_index_positions[
                    0
                ],
            ],
            events.__getitem__(self.config.root_ntuple_config.JetConfig.pt),
        )[0]
        jet_truth_3 = ak.broadcast_arrays(
            jet_truth_padded[
                :,
                self.config.root_ntuple_config.MatchingConfig.jet_parton_match_index_positions[
                    1
                ],
            ],
            events.__getitem__(self.config.root_ntuple_config.JetConfig.pt),
        )[0]
        sort_idx = ak.argsort(
            events.__getitem__(self.config.root_ntuple_config.JetConfig.pt),
            ascending=False,
        )

        jet_pt = events.__getitem__(self.config.root_ntuple_config.JetConfig.pt)[
            sort_idx
        ]
        jet_eta = events.__getitem__(self.config.root_ntuple_config.JetConfig.eta)[
            sort_idx
        ]
        jet_phi = events.__getitem__(self.config.root_ntuple_config.JetConfig.phi)[
            sort_idx
        ]
        jet_e = events.__getitem__(self.config.root_ntuple_config.JetConfig.energy)[
            sort_idx
        ]
        jet_btag = events.__getitem__(self.config.root_ntuple_config.JetConfig.btag)[
            sort_idx
        ]
        jet_truth = ak.where(
            jet_idx == jet_truth_0, 1, ak.where(jet_idx == jet_truth_3, -1, 0)
        )[sort_idx]
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
        jet_btag_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_btag, max_jets, clip=True), self.config.padding_value
            )
        )
        jet_truth_np = ak.to_numpy(
            ak.fill_none(
                ak.pad_none(jet_truth, max_jets, clip=True), self.config.padding_value
            )
        )

        event_jet_truth_idx = np.full((n_events, 6), -1, dtype=np.int32)
        for idx in range(max_jets):
            mask_top = jet_truth_np[:, idx] == 1
            mask_tbar = jet_truth_np[:, idx] == -1
            event_jet_truth_idx[mask_top, 0] = idx
            event_jet_truth_idx[mask_tbar, 3] = idx

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
            "event_jet_truth_idx": event_jet_truth_idx,
            "N_jets": n_jets,
            "N_bjets": n_bjets,
        }

    def _process_met(self, events: ak.Array) -> Dict[str, np.ndarray]:
        met_met = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.METConfig.met_met)
        )
        met_phi = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.METConfig.met_phi)
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

    def _extract_truth_info(self, events: ak.Array) -> Dict[str, np.ndarray]:
        """
        Extract truth-level information.

        Args:
            events: Event array

        Returns:
            Dictionary of truth features
        """

        # Extract truth top/anti-top 4-vectors
        truth_top_pt = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.top_pt)
        )
        truth_top_eta = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.top_eta)
        )
        truth_top_phi = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.top_phi)
        )
        truth_top_mass = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.top_mass)
        )

        truth_tbar_pt = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.tbar_pt)
        )
        truth_tbar_eta = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.tbar_eta)
        )
        truth_tbar_phi = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.tbar_phi)
        )
        truth_tbar_mass = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.tbar_mass)
        )

        # Compute ttbar system
        top_px = truth_top_pt * np.cos(truth_top_phi)
        top_py = truth_top_pt * np.sin(truth_top_phi)
        top_pz = truth_top_pt * np.sinh(truth_top_eta)
        top_e = np.sqrt(truth_top_mass**2 + top_px**2 + top_py**2 + top_pz**2)

        tbar_px = truth_tbar_pt * np.cos(truth_tbar_phi)
        tbar_py = truth_tbar_pt * np.sin(truth_tbar_phi)
        tbar_pz = truth_tbar_pt * np.sinh(truth_tbar_eta)
        tbar_e = np.sqrt(truth_tbar_mass**2 + tbar_px**2 + tbar_py**2 + tbar_pz**2)

        ttbar_e = top_e + tbar_e
        ttbar_px = top_px + tbar_px
        ttbar_py = top_py + tbar_py
        ttbar_pz = top_pz + tbar_pz

        truth_ttbar_mass = np.sqrt(ttbar_e**2 - ttbar_px**2 - ttbar_py**2 - ttbar_pz**2)
        truth_ttbar_pt = np.sqrt(ttbar_px**2 + ttbar_py**2)
        ttbar_p = np.sqrt(ttbar_px**2 + ttbar_py**2 + ttbar_pz**2)
        truth_tt_boost_parameter = ttbar_p / ttbar_e

        # Lepton 4-vectors from W decay
        lep_top_pt = ak.to_numpy(
            events.__getitem__(self.config.root_ntuple_config.TruthConfig.top_lepton_pt)
        )
        lep_top_eta = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_lepton_eta
            )
        )
        lep_top_phi = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_lepton_phi
            )
        )
        lep_top_mass = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_lepton_mass
            )
        )
        lep_top_e = np.sqrt(lep_top_mass**2 + lep_top_pt**2 * np.cosh(lep_top_eta) ** 2)

        # Lepton from anti-top
        lep_tbar_pt = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_lepton_pt
            )
        )
        lep_tbar_eta = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_lepton_eta
            )
        )
        lep_tbar_phi = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_lepton_phi
            )
        )
        lep_tbar_mass = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_lepton_mass
            )
        )
        lep_tbar_e = np.sqrt(
            lep_tbar_mass**2 + lep_tbar_pt**2 * np.cosh(lep_tbar_eta) ** 2
        )
        top_lep_px = lep_top_pt * np.cos(lep_top_phi)
        top_lep_py = lep_top_pt * np.sin(lep_top_phi)
        top_lep_pz = lep_top_pt * np.sinh(lep_top_eta)

        tbar_lep_px = lep_tbar_pt * np.cos(lep_tbar_phi)
        tbar_lep_py = lep_tbar_pt * np.sin(lep_tbar_phi)
        tbar_lep_pz = lep_tbar_pt * np.sinh(lep_tbar_eta)

        # Extract neutrino information
        nu_top_pt = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_neutrino_pt
            )
        )
        nu_top_eta = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_neutrino_eta
            )
        )
        nu_top_phi = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_neutrino_phi
            )
        )
        nu_top_mass = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.top_neutrino_mass
            )
        )

        nu_tbar_pt = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_neutrino_pt
            )
        )
        nu_tbar_eta = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_neutrino_eta
            )
        )
        nu_tbar_phi = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_neutrino_phi
            )
        )
        nu_tbar_mass = ak.to_numpy(
            events.__getitem__(
                self.config.root_ntuple_config.TruthConfig.tbar_neutrino_mass
            )
        )

        # Compute neutrino px, py, pz
        nu_top_px = nu_top_pt * np.cos(nu_top_phi)
        nu_top_py = nu_top_pt * np.sin(nu_top_phi)
        nu_top_pz = nu_top_pt * np.sinh(nu_top_eta)
        nu_top_e = np.sqrt(nu_top_mass**2 + nu_top_px**2 + nu_top_py**2 + nu_top_pz**2)

        nu_tbar_px = nu_tbar_pt * np.cos(nu_tbar_phi)
        nu_tbar_py = nu_tbar_pt * np.sin(nu_tbar_phi)
        nu_tbar_pz = nu_tbar_pt * np.sinh(nu_tbar_eta)
        nu_tbar_e = np.sqrt(
            nu_tbar_mass**2 + nu_tbar_px**2 + nu_tbar_py**2 + nu_tbar_pz**2
        )

        truth_c_hel = c_hel(
            np.stack([top_px, top_py, top_pz, top_e], axis=1),
            np.stack([tbar_px, tbar_py, tbar_pz, tbar_e], axis=1),
            np.stack([top_lep_px, top_lep_py, top_lep_pz, lep_top_e], axis=1),
            np.stack([tbar_lep_px, tbar_lep_py, tbar_lep_pz, lep_tbar_e], axis=1),
        )

        truth_c_han = c_han(
            np.stack([top_px, top_py, top_pz, top_e], axis=1),
            np.stack([tbar_px, tbar_py, tbar_pz, tbar_e], axis=1),
            np.stack([top_lep_px, top_lep_py, top_lep_pz, lep_top_e], axis=1),
            np.stack([tbar_lep_px, tbar_lep_py, tbar_lep_pz, lep_tbar_e], axis=1),
        )

        truth_c_hel = np.clip(truth_c_hel, -1, 1)
        truth_c_han = np.clip(truth_c_han, -1, 1)

        return {
            # ttbar system
            "truth_ttbar_mass": truth_ttbar_mass,
            "truth_ttbar_pt": truth_ttbar_pt,
            "truth_tt_boost_parameter": truth_tt_boost_parameter,
            "truth_ttbar_px": ttbar_px,
            "truth_ttbar_py": ttbar_py,
            "truth_ttbar_pz": ttbar_pz,
            "truth_ttbar_e": ttbar_e,
            "truth_ttbar_p": ttbar_p,
            # Top quark
            "truth_top_mass": truth_top_mass,
            "truth_top_pt": truth_top_pt,
            "truth_top_eta": truth_top_eta,
            "truth_top_phi": truth_top_phi,
            "truth_top_e": top_e,
            # Anti-top quark
            "truth_tbar_mass": truth_tbar_mass,
            "truth_tbar_pt": truth_tbar_pt,
            "truth_tbar_eta": truth_tbar_eta,
            "truth_tbar_phi": truth_tbar_phi,
            "truth_tbar_e": tbar_e,
            # Neutrinos
            "truth_top_neutino_mass": nu_top_mass,
            "truth_top_neutino_pt": nu_top_pt,
            "truth_top_neutino_eta": nu_top_eta,
            "truth_top_neutino_phi": nu_top_phi,
            "truth_top_neutino_e": nu_top_e,
            "truth_top_neutrino_px": nu_top_px,
            "truth_top_neutrino_py": nu_top_py,
            "truth_top_neutrino_pz": nu_top_pz,
            "truth_tbar_neutino_mass": nu_tbar_mass,
            "truth_tbar_neutino_pt": nu_tbar_pt,
            "truth_tbar_neutino_eta": nu_tbar_eta,
            "truth_tbar_neutino_phi": nu_tbar_phi,
            "truth_tbar_neutino_e": nu_tbar_e,
            "truth_tbar_neutrino_px": nu_tbar_px,
            "truth_tbar_neutrino_py": nu_tbar_py,
            "truth_tbar_neutrino_pz": nu_tbar_pz,
            # Leptons from W decays
            "truth_top_lepton_e": lep_top_e,
            "truth_tbar_lepton_e": lep_tbar_e,
            "truth_top_lepton_px": top_lep_px,
            "truth_top_lepton_py": top_lep_py,
            "truth_top_lepton_pz": top_lep_pz,
            "truth_tbar_lepton_px": tbar_lep_px,
            "truth_tbar_lepton_py": tbar_lep_py,
            "truth_tbar_lepton_pz": tbar_lep_pz,
            # Truth c_han, c_hel
            "truth_c_hel": truth_c_hel,
            "truth_c_han": truth_c_han,
        }

    def _extract_neutrino_reco(self, events: ak.Array) -> Dict[str, np.ndarray]:
        """
        Extract neutrino reconstruction results.

        Args:
            events: Event array

        Returns:
            Dictionary of neutrino reconstruction features
        """
        output = {}
        if self.config.root_ntuple_config.NeutrinoReco is None:
            return output
        for reco_config in self.config.root_ntuple_config.NeutrinoReco:

            if isinstance(reco_config.nu_px, str):
                nu_px = ak.to_numpy(events.__getitem__(reco_config.nu_px))
                nu_py = ak.to_numpy(events.__getitem__(reco_config.nu_py))
                nu_pz = ak.to_numpy(events.__getitem__(reco_config.nu_pz))
                nubar_px = ak.to_numpy(events.__getitem__(reco_config.nubar_px))
                nubar_py = ak.to_numpy(events.__getitem__(reco_config.nubar_py))
                nubar_pz = ak.to_numpy(events.__getitem__(reco_config.nubar_pz))
            elif (
                isinstance(reco_config.nu_px, int)
                and reco_config.branch_name is not None
            ):
                nu_px = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[..., reco_config.nu_px]
                )
                nu_py = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[..., reco_config.nu_py]
                )
                nu_pz = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[..., reco_config.nu_pz]
                )
                nubar_px = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[
                        ..., reco_config.nubar_px
                    ]
                )
                nubar_py = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[
                        ..., reco_config.nubar_py
                    ]
                )
                nubar_pz = ak.to_numpy(
                    events.__getitem__(reco_config.branch_name)[
                        ..., reco_config.nubar_pz
                    ]
                )
            else:
                raise ValueError(
                    f"Invalid neutrino reconstruction configuration: {reco_config}"
                )
            output[f"{reco_config.name}_nu_px"] = nu_px * 1e3
            output[f"{reco_config.name}_nu_py"] = nu_py * 1e3
            output[f"{reco_config.name}_nu_pz"] = nu_pz * 1e3
            output[f"{reco_config.name}_nubar_px"] = nubar_px * 1e3
            output[f"{reco_config.name}_nubar_py"] = nubar_py * 1e3
            output[f"{reco_config.name}_nubar_pz"] = nubar_pz * 1e3
        return output

    def save_to_npz(self, output_path: str):
        """Save data to NPZ format."""
        np.savez_compressed(output_path, **self.processed_data)

    def save_to_root(self, output_path):
        """Save data to ROOT format."""
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Convert arrays to appropriate format for uproot
        output_dict = {}
        for key, value in self.processed_data.items():
            if value.ndim == 1:
                # Scalar branches
                output_dict[key] = value
            else:
                # Vector branches - convert to jagged arrays
                output_dict[key] = ak.Array([value[i] for i in range(len(value))])

        # Write to ROOT file
        with uproot.recreate(output_path) as file:
            file[self.config.tree_name] = output_dict

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


def preprocess_root_file(
    input_path: str,
    output_path: str,
    tree_name: str = "reco",
    output_format: str = "root",
    save_initial_parton_info: bool = True,
    verbose: bool = True,
    max_jets: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to preprocess a ROOT file.

    Args:
        input_path: Path to input ROOT file
        output_path: Path to output file
        tree_name: Name of tree in ROOT file
        output_format: Output format ('root' or 'npz')
        save_nu_flows: Whether to save NuFlow results
        save_initial_parton_info: Whether to save initial parton info
        verbose: Whether to print progress

    Returns:
        Dictionary of processed data
    """
    config = configs.PreprocessorConfig(
        input_path=input_path,
        tree_name=tree_name,
        save_initial_parton_info=save_initial_parton_info,
        verbose=verbose,
        max_saved_jets=max_jets,
    )

    preprocessor = RootPreprocessor(config)
    preprocessor.process()
    if output_format == "npz":
        preprocessor.save_to_npz(output_path)
    elif output_format == "root":
        preprocessor.save_to_root(output_path)

    return preprocessor.get_processed_data()


def preprocess_root_directory(
    config: DataSampleConfig,
):
    """
    Process all ROOT files in a directory.

    Args:
        input_dir: Directory containing input ROOT files
        output_dir: Directory to save processed files
        tree_name: Name of tree in ROOT files
        output_format: Output format ('root' or 'npz')
        save_nu_flows: Whether to save NuFlow results
        save_initial_parton_info: Whether to save initial parton info
        verbose: Whether to print progress
        max_jets: Maximum number of jets to save
        num_events: Maximum number of events to process (None for all)
    """
    data_collected = []
    input_dir = config.input_dir
    output_file = os.path.join(config.output_dir, config.name + ".npz")
    num_events = config.num_events

    num_total_events = 0
    preprocessor = RootPreprocessor(config.preprocessor_config)

    root_files = (
        [
            os.path.join(input_dir, filename)
            for filename in sorted(os.listdir(input_dir))
            if filename.endswith(".root")
        ]
        if os.path.isdir(input_dir)
        else [input_dir]
    )
    num_files = len(root_files)
    print(f"Found {num_files} files in {input_dir}.\nStarting processing...\n\n")


    for file_index, input_path in enumerate(root_files):
        print(f"Processing file {file_index + 1} of {num_files}...\n")
        if input_path.endswith(".root"):

            data_collected.append(preprocessor.process(input_path))
            num_events = preprocessor.get_num_events()
            num_total_events += num_events
            print(
                f"Processed {num_events} events. Total events so far: {num_total_events}\n\n"
            )
        if num_events is not None and num_total_events >= num_events:
            print(
                f"Reached maximum number of events: {num_events}. Stopping processing."
            )
            break
    print(f"Finished processing files. Total events processed: {num_total_events}")
    print("\nMerging data from all files...")
    merged_data = {}
    n_keys = len(data_collected[0].keys())
    for key in data_collected[0].keys():
        print(f"Merging key: {key} ({len(merged_data)+1}/{n_keys})")
        for data in data_collected:
            if key not in data:
                raise KeyError(
                    f"Key '{key}' not found in one of the data dictionaries."
                )
        merged_data[key] = np.concatenate(
            [data[key] for data in data_collected], axis=0
        )

    print("Saving merged data to npz file...")
    np.savez_compressed(output_file, **merged_data)
    print(f"Merged data saved to {output_file}")
