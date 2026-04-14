from ..base_classes.reconstruction_base import EventReconstructorBase
from core.DataLoader import DataConfig
import numpy as np
from core.utils.four_vector_arithmetics import (
    lorentz_vector_from_pt_eta_phi_e,
    compute_mass_from_lorentz_vector,
)


class BaselineAssigner(EventReconstructorBase):
    def __init__(
        self,
        config: DataConfig,
        name="baseline_assigner",
        mode="min",
        use_nu_flows=False,
        manchester_style=False,
    ):
        super().__init__(
            config,
            assignment_name=name,
            full_reco_name=name
            + (r"+ $\nu^2$-Flows" if use_nu_flows else r" + True $\nu$"),
            perform_regression=False,
            use_nu_flows=use_nu_flows,
        )
        """Initializes the BaselineAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
            mode (str): The mode of assignment, either "min" or "max".
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")
        self.mode = mode
        self.NUM_LEPTONS = config.NUM_LEPTONS
        self.max_jets = config.max_jets
        self.padding_value = config.padding_value
        self.lepton_inputs = config.lepton_inputs
        self.jet_inputs = config.jet_inputs
        self.feature_index_dict = config.feature_indices
        self.b_tag_threshold = 2
        self.manchester_style = manchester_style

    def compute_comparison_feature(self, data_dict):
        """
        Computes a comparison feature between leptons and jets.
        This is a placeholder implementation and should be replaced with actual logic.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets,NUM_LEPTONS) representing the comparison feature.
        """
        raise NotImplementedError(
            "This method should be implemented with actual logic."
        )

    def get_jets_mask(self, data_dict):
        padding_value = self.config.padding_value
        jet_mask = (data_dict["jet_inputs"] != padding_value).all(
            axis=-1
        )  # Shape: (num_events, max_jets)

        jet_mask = np.repeat(jet_mask[:, :, np.newaxis], self.NUM_LEPTONS, axis=2)
        return jet_mask

    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton based on the padding value.
        Selects jets with b-tag score >= 2. If there are fewer than 2 such jets,
        fills up with the pt-leading non-tagged jets.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_jets, NUM_LEPTONS)
                        indicating viable jets.
        """
        jet_mask = self.get_jets_mask(data_dict)
        jet_b_tag = None
        jet_pt = None

        for feature in self.feature_index_dict["jet_inputs"]:
            if "btag" in feature.lower() or "b_tag" in feature.lower():
                jet_b_tag = data_dict["jet_inputs"][
                    :, :, self.feature_index_dict["jet_inputs"][feature]
                ]
            if "pt" in feature.lower():
                jet_pt = data_dict["jet_inputs"][
                    :, :, self.feature_index_dict["jet_inputs"][feature]
                ]

        if jet_b_tag is not None:
            b_tag_mask = (jet_b_tag >= 2) & jet_mask[:, :, 0]

            less_than_2_b_tag_jet_mask = np.sum(b_tag_mask, axis=1) < 2
            more_than_2_b_tag_jet_mask = np.sum(b_tag_mask, axis=1) > 2

            # For more than 2 b-tagged jets, keep only the leading 2 by pt
            if more_than_2_b_tag_jet_mask.any():
                b_tagged_jet_pt = jet_pt * b_tag_mask
                second_leading_b_tagged_jet_pt = np.partition(
                    b_tagged_jet_pt[more_than_2_b_tag_jet_mask], -2, axis=1
                )[:, -2][:, np.newaxis]
                b_tag_mask[more_than_2_b_tag_jet_mask] &= (
                    b_tagged_jet_pt[more_than_2_b_tag_jet_mask]
                    >= second_leading_b_tagged_jet_pt
                )
            if self.manchester_style:
                # For Manchester-style, if there are less than 2 b-tagged jets, keep all jets (including non-tagged) as viable
                b_tag_mask[less_than_2_b_tag_jet_mask] = jet_mask[
                    less_than_2_b_tag_jet_mask
                ][:, :, 0]
            else:
                # For less than 2 b-tagged jets, fill up with leading non-tagged jets
                iteration = 0
                while less_than_2_b_tag_jet_mask.any():
                    jet_pt_masked = jet_pt.copy()
                    jet_pt_masked[b_tag_mask] = -1

                    leading_jet_pt_indices = np.argmax(jet_pt_masked, axis=1)
                    leading_jet_pt_indices = leading_jet_pt_indices[:, np.newaxis]

                    update_mask = (
                        leading_jet_pt_indices[less_than_2_b_tag_jet_mask]
                        == np.arange(self.max_jets)[np.newaxis, :]
                    )
                    b_tag_mask[less_than_2_b_tag_jet_mask] |= update_mask
                    less_than_2_b_tag_jet_mask = np.sum(b_tag_mask, axis=1) < 2

                    iteration += 1
                    if iteration > self.max_jets:
                        raise RuntimeError(
                            "Exceeded maximum iterations while ensuring at least 2 b-tagged jets."
                        )
                jet_mask = jet_mask & b_tag_mask[:, :, np.newaxis]
        return jet_mask

    def predict_indices(self, data_dict):
        """
        Predicts the indices of jets assigned to each lepton based on a comparison feature.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets,NUM_LEPTONS) representing the predicted indices.
        """
        comparison_feature = self.compute_comparison_feature(data_dict)
        viable_jets_mask = self.get_viable_jets_mask(data_dict)
        num_events = comparison_feature.shape[0]
        predicted_indices = np.zeros((num_events, self.max_jets, self.NUM_LEPTONS))
        for i in range(num_events):
            for j in range(self.NUM_LEPTONS):
                valid_indices = np.where(viable_jets_mask[i, :, j])[0]
                if valid_indices.size > 0:
                    if self.mode == "min":
                        best_jet_index = valid_indices[
                            np.argmin(comparison_feature[i, valid_indices, j])
                        ]
                    else:  # self.mode == "max"
                        best_jet_index = valid_indices[
                            np.argmax(comparison_feature[i, valid_indices, j])
                        ]
                    predicted_indices[i, best_jet_index, j] = 1
                    viable_jets_mask[i, best_jet_index, :] = (
                        False  # Exclude this jet for other leptons
                    )
        return predicted_indices


class DeltaRAssigner(BaselineAssigner):
    def __init__(self, config: DataConfig, name=None, **kwargs):
        super().__init__(
            config,
            name=(name if name is not None else r"$\Delta R(\ell,j)$-Method"),
            **kwargs
        )
        """Initializes the DeltaRAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
            mode (str): The mode of assignment, either "min" or "max".
        """

    def compute_comparison_feature(self, data_dict):
        """
        Computes the Delta R between leptons and jets.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets, NUM_LEPTONS) representing the Delta R values.
        """
        leptons = data_dict["lep_inputs"]
        jets = data_dict["jet_inputs"]
        lepton_eta = None
        lepton_phi = None
        for feature in self.lepton_inputs:
            if "eta" in feature.lower():
                lepton_eta = leptons[
                    :, :, self.feature_index_dict["lep_inputs"][feature]
                ]
            if "phi" in feature.lower():
                lepton_phi = leptons[
                    :, :, self.feature_index_dict["lep_inputs"][feature]
                ]
        jet_eta = None
        jet_phi = None
        for feature in self.jet_inputs:
            if "eta" in feature.lower():
                jet_eta = jets[:, :, self.feature_index_dict["jet_inputs"][feature]]
            if "phi" in feature.lower():
                jet_phi = jets[:, :, self.feature_index_dict["jet_inputs"][feature]]
        if (
            lepton_eta is None
            or lepton_phi is None
            or jet_eta is None
            or jet_phi is None
        ):
            raise ValueError(
                "Eta and Phi features must be present in both leptons and jets."
            )
        delta_eta = jet_eta[:, :, np.newaxis] - lepton_eta[:, np.newaxis, :]
        delta_phi = jet_phi[:, :, np.newaxis] - lepton_phi[:, np.newaxis, :]
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap-around
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_r


class ChiSquareAssigner(BaselineAssigner):
    """Assigns jets to leptons based on mass combinatorics involving neutrino momenta."""

    def __init__(
        self,
        config: DataConfig,
        use_nu_flows_for_assignment=True,
        top_mass=173.15e3,
        **kwargs
    ):
        super().__init__(config, name=((r"$\chi^2$-Method")), **kwargs)
        """Initializes the ChiSquareAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
        """
        self.NUM_LEPTONS = config.NUM_LEPTONS
        self.feature_index_dict = config.feature_indices
        self.max_jets = config.max_jets
        self.top_mass = top_mass
        self.use_nu_flows_for_assignment = use_nu_flows_for_assignment

    def get_neutrino_momenta(self, data_dict):
        if self.use_nu_flows_for_assignment:
            return data_dict["nu_flows_neutrino_truth"]
        else:
            return data_dict["regression"]

    def construct_neutrino_four_vectors(self, data_dict):
        neutrino_3_vector = self.get_neutrino_momenta(data_dict)
        nu_e = np.linalg.norm(neutrino_3_vector[:, 0, :], axis=-1)
        anu_e = np.linalg.norm(neutrino_3_vector[:, 1, :], axis=-1)
        nu_four_vector = np.concatenate(
            [neutrino_3_vector[:, 0, :], nu_e[:, np.newaxis]], axis=-1
        )
        anu_four_vector = np.concatenate(
            [neutrino_3_vector[:, 1, :], anu_e[:, np.newaxis]], axis=-1
        )
        return nu_four_vector, anu_four_vector

    def get_invariant_mass(self, four_vector):
        px, py, pz, e = (
            four_vector[..., 0],
            four_vector[..., 1],
            four_vector[..., 2],
            four_vector[..., 3],
        )
        mass_squared = e**2 - (px**2 + py**2 + pz**2)
        mass_squared = np.maximum(
            mass_squared, 0
        )  # Avoid negative values due to numerical issues
        return np.sqrt(mass_squared)

    def get_four_vector(self, pt, eta, phi, e):
        px, py, pz, energy = lorentz_vector_from_pt_eta_phi_e(pt, eta, phi, e)
        return np.array([px, py, pz, energy]).transpose(
            1, 2, 0
        )  # Shape: (num_events, num_particles,4)

    def predict_indices(self, data_dict):
        """
        Predicts the indices of jets assigned to each lepton based on mass combinatorics.
        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets, NUM_LEPTONS) representing the predicted indices.
        """
        leptons = data_dict["lep_inputs"]
        jets = data_dict["jet_inputs"][
            :, :, :4
        ]  # Assuming first 4 features are pt, eta, phi, e

        lepton_four_vectors = self.get_four_vector(
            leptons[:, :, self.feature_index_dict["lep_inputs"]["lep_pt"]],
            leptons[:, :, self.feature_index_dict["lep_inputs"]["lep_eta"]],
            leptons[:, :, self.feature_index_dict["lep_inputs"]["lep_phi"]],
            leptons[:, :, self.feature_index_dict["lep_inputs"]["lep_e"]],
        )

        jet_four_vectors = self.get_four_vector(
            jets[:, :, self.feature_index_dict["jet_inputs"]["jet_pt"]],
            jets[:, :, self.feature_index_dict["jet_inputs"]["jet_eta"]],
            jets[:, :, self.feature_index_dict["jet_inputs"]["jet_phi"]],
            jets[:, :, self.feature_index_dict["jet_inputs"]["jet_e"]],
        )

        nu_four_vector, anu_four_vector = self.construct_neutrino_four_vectors(
            data_dict
        )
        viable_jets_mask = self.get_viable_jets_mask(
            data_dict
        )  # Shape: (num_events, max_jets, NUM_LEPTONS)
        W_boson_candidates = lepton_four_vectors + np.stack(
            [nu_four_vector, anu_four_vector], axis=1
        )  # Shape: (num_events, NUM_LEPTONS,4)
        W_boson_candidates = W_boson_candidates[:, np.newaxis, :, :]
        top_candidates = W_boson_candidates + jet_four_vectors[:, :, np.newaxis, :]
        # Shape: (num_events, max_jets, NUM_LEPTONS,4)

        top_masses = self.get_invariant_mass(
            top_candidates
        )  # Shape: (num_events, max_jets, NUM_LEPTONS)
        mass_differences = np.square(
            top_masses - self.top_mass
        )  # Shape: (num_events, max_jets, NUM_LEPTONS)
        mass_differences_masked = np.where(
            viable_jets_mask, mass_differences, np.inf
        )  # Shape: (num_events, max_jets, NUM_LEPTONS)
        num_events = mass_differences_masked.shape[0]
        predicted_indices = np.zeros((num_events, self.max_jets, self.NUM_LEPTONS))
        from scipy.optimize import linear_sum_assignment

        for e in range(num_events):
            cost_matrix = mass_differences_masked[e].T  # Shape: (NUM_LEPTONS, max_jets)
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except ValueError:
                print("WARNING: Linear sum assignment failed for event", e)
                row_ind, col_ind = [], []
                continue
            predicted_indices[e, col_ind, row_ind] = 1

        return predicted_indices
