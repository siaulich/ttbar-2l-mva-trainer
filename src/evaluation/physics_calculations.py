"""Physics calculations for event reconstruction."""

import numpy as np
from typing import Tuple, Optional, Callable

from ..utils import (
    lorentz_vector_from_PtEtaPhiE_array,
    lorentz_vector_from_neutrino_momenta_array,
    compute_mass_from_lorentz_vector_array,
)


class ResolutionCalculator:
    """Calculate mass resolution metrics."""

    @staticmethod
    def compute_deviation(
        reco_values: np.ndarray,
        true_values: np.ndarray,
        use_relative_deviation: bool = True,
        use_signed_deviation: bool = True,
        deviation_function: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Compute deviations between reconstructed and true masses.

        Args:
            reco_values: Array of reconstructed masses
            true_values: Array of true masses
            use_relative_deviation: If True, compute relative deviations
            use_signed_deviation: If True, keep sign of deviations
        Returns:
            Array of deviations
        """
        # Filter out invalid values before computation
        valid_mask = (
            np.isfinite(reco_values)
            & np.isfinite(true_values)
            & (true_values != 0)
            & (reco_values != -999)  # padding value
        )

        if deviation_function is not None:
            deviations = deviation_function(true_values, reco_values)
        elif use_relative_deviation:
            with np.errstate(divide="ignore", invalid="ignore"):
                deviations = (reco_values - true_values) / true_values
            if not use_signed_deviation:
                deviations = np.abs(deviations)
        else:
            deviations = reco_values - true_values
            if not use_signed_deviation:
                deviations = np.abs(deviations)

        # Set invalid entries to NaN so they can be filtered later
        deviations = np.where(valid_mask, deviations, np.nan)

        return deviations

    @staticmethod
    def compute_resolution(
        deviations: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute resolution as standard deviation of deviations.

        Args:
            deviations: Array of relative deviations
            weights: Optional event weights

        Returns:
            Resolution value
        """
        if weights is None:
            weights = np.ones_like(deviations)

        mean_deviation = np.average(deviations, weights=weights)
        variance = np.average(
            (deviations - mean_deviation) ** 2,
            weights=weights,
        )
        return np.sqrt(variance)

    @staticmethod
    def compute_binned_resolution(
        deviations: np.ndarray,
        binning_mask: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute resolution in bins.

        Args:
            deviations: Array of relative deviations
            binning_mask: Boolean mask for binning (n_bins, n_events)
            weights: Event weights

        Returns:
            Array of resolutions per bin
        """
        n_bins = binning_mask.shape[0]
        resolutions = np.zeros(n_bins)

        for i in range(n_bins):
            bin_mask = binning_mask[i]
            if np.sum(bin_mask) > 0:
                bin_deviations = deviations[bin_mask]
                bin_weights = weights[bin_mask]
                resolutions[i] = ResolutionCalculator.compute_resolution(
                    bin_deviations, bin_weights
                )

        return resolutions


import numpy as np


# ----------------------------------------------------------------------
# Lorentz boost of v into the rest frame of parent, both shaped (N,4)
# Components are (px, py, pz, E)
# ----------------------------------------------------------------------
def boost(v, parent):
    """
    Boost 4-vectors v into the rest frame of parent.
    v, parent: shape (N,4), order (px,py,pz,E)
    Returns boosted (N,4) in the same order.
    """
    pv = parent[:, :3]
    Ep = parent[:, 3]

    # Beta points along parent momentum; negate to boost to rest frame
    p = np.linalg.norm(pv, axis=1)

    # Safe beta calculation
    beta = np.zeros_like(pv)
    mask = (Ep > 1e-10) & (p > 0) & np.isfinite(Ep) & np.all(np.isfinite(pv), axis=1)
    beta[mask] = pv[mask] / Ep[mask, None]

    # Gamma factor with safe division and overflow protection
    beta_sq = np.sum(beta**2, axis=1)
    beta_sq = np.clip(beta_sq, 0, 1 - 1e-15)  # Ensure < 1 for physical values
    gamma = 1.0 / np.sqrt(np.maximum(1 - beta_sq, 1e-10))
    gamma = np.clip(gamma, 1.0, 1e6)  # Limit gamma to reasonable values

    vv = v[:, :3]
    E = v[:, 3]

    # Boost formulas (negating beta for rest frame boost)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_dot_v = np.sum(beta * vv, axis=1)

        # Safe computation of boost components
        denominator = np.maximum(beta_sq[:, None], 1e-10)
        boost_correction = (
            (gamma[:, None] - 1) * beta_dot_v[:, None] * beta / denominator
        )

        v_prime = vv - gamma[:, None] * beta * E[:, None] + boost_correction
        E_prime = gamma * (E - beta_dot_v)

    # Replace any NaN or Inf with original values
    valid = np.isfinite(v_prime).all(axis=1) & np.isfinite(E_prime)
    v_prime = np.where(valid[:, None], v_prime, v[:, :3])
    E_prime = np.where(valid, E_prime, v[:, 3])

    return np.column_stack([v_prime, E_prime])


# ----------------------------------------------------------------------
# Common boost sequence (same for cos_han and c_hel)
# ----------------------------------------------------------------------
def _prep_leptons(top, tbar, lep_pos, lep_neg):
    """
    Apply:
      1) boost everything to ttbar rest frame
      2) boost leptons into top / tbar rest frames
    Returns (lep_pos_3vec, lep_neg_3vec)
    """

    # --- ttbar 4-vector
    ttbar = np.zeros_like(top)
    ttbar[:, :3] = top[:, :3] + tbar[:, :3]
    ttbar[:, 3] = top[:, 3] + tbar[:, 3]

    lep_pos_1 = boost(lep_pos, ttbar)
    lep_neg_1 = boost(lep_neg, ttbar)
    top_1 = boost(top, ttbar)
    tbar_1 = boost(tbar, ttbar)

    lep_pos_2 = boost(lep_pos_1, top_1)
    lep_neg_2 = boost(lep_neg_1, tbar_1)

    return lep_pos_2[:, :3], lep_neg_2[:, :3]


def select_jets(jet_inputs, assignment_pred):
    selected_jet_indices = assignment_pred.argmax(axis=-2)
    reco_jets = np.take_along_axis(
        jet_inputs,
        selected_jet_indices[:, :, np.newaxis],
        axis=1,
    )
    return reco_jets


# ----------------------------------------------------------------------
# cos_han  (CMS version: flip neg lepton z-component)
# ----------------------------------------------------------------------
def c_han(top, tbar, lep_pos, lep_neg):
    p1, p2 = _prep_leptons(top, tbar, lep_pos, lep_neg)

    # CMS-specific flip along tbar direction
    with np.errstate(divide="ignore", invalid="ignore"):
        tbar_norm = np.linalg.norm(tbar[:, :3], axis=1, keepdims=True)
        tbar_norm = np.clip(tbar_norm, 1e-10, None)  # Avoid division by zero

        # Check for finite values
        valid_tbar = np.isfinite(tbar_norm) & np.all(
            np.isfinite(tbar[:, :3]), axis=1, keepdims=True
        )
        tbar_norm = np.where(valid_tbar, tbar_norm, 1.0)

        tbar_dir = tbar[:, :3] / tbar_norm
        tbar_dir = np.where(
            valid_tbar, tbar_dir, np.array([0, 0, 1])
        )  # Default z-direction

        # Project and flip p2 along tbar direction
        p2_z = np.sum(p2 * tbar_dir, axis=1)
        p2_z_inverted = p2 - 2 * (p2_z[:, None] * tbar_dir)

        # Safely handle p2_z_inverted if p2 was invalid
        valid_p2 = np.all(np.isfinite(p2), axis=1, keepdims=True)
        p2_z_inverted = np.where(valid_p2, p2_z_inverted, p2)

        # Normalize with safe division
        p1_norm = np.linalg.norm(p1, axis=1, keepdims=True)
        p1_norm = np.clip(p1_norm, 1e-10, None)
        valid_p1 = np.isfinite(p1_norm) & np.all(np.isfinite(p1), axis=1, keepdims=True)
        p1_norm = np.where(valid_p1, p1_norm, 1.0)
        u1 = p1 / p1_norm
        u1 = np.where(valid_p1, u1, 0.0)

        p2_inv_norm = np.linalg.norm(p2_z_inverted, axis=1, keepdims=True)
        p2_inv_norm = np.clip(p2_inv_norm, 1e-10, None)
        valid_p2_inv = np.isfinite(p2_inv_norm) & np.all(
            np.isfinite(p2_z_inverted), axis=1, keepdims=True
        )
        p2_inv_norm = np.where(valid_p2_inv, p2_inv_norm, 1.0)
        u2 = p2_z_inverted / p2_inv_norm
        u2 = np.where(valid_p2_inv, u2, 0.0)

        result = np.sum(u1 * u2, axis=1)
        # Return 0 for invalid cases
        result = np.where(np.isfinite(result), result, 0.0)

    return result


# ----------------------------------------------------------------------
# c_hel (standard helicity angle: NO flipping)
# ----------------------------------------------------------------------
def c_hel(top, tbar, lep_pos, lep_neg):
    p1, p2 = _prep_leptons(top, tbar, lep_pos, lep_neg)

    p1_norm = np.linalg.norm(p1, axis=1, keepdims=True)
    p1_norm = np.maximum(p1_norm, 1e-10)
    u1 = p1 / p1_norm

    p2_norm = np.linalg.norm(p2, axis=1, keepdims=True)
    p2_norm = np.maximum(p2_norm, 1e-10)
    u2 = p2 / p2_norm

    return np.sum(u1 * u2, axis=1)


def c_hel_test(top, tbar, lep_pos, lep_neg):
    lep_pos_boosted = boost(lep_pos, top)
    lep_neg_boosted = boost(lep_neg, tbar)
    p1 = lep_pos_boosted[:, :3]
    p2 = lep_neg_boosted[:, :3]
    p1_norm = np.linalg.norm(p1, axis=1, keepdims=True)
    p1_norm = np.maximum(p1_norm, 1e-10)
    u1 = p1 / p1_norm
    p2_norm = np.linalg.norm(p2, axis=1, keepdims=True)
    p2_norm = np.maximum(p2_norm, 1e-10)
    u2 = p2 / p2_norm
    return np.sum(u1 * u2, axis=1)
