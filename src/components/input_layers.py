import keras as keras
import tensorflow as tf
import math


@keras.utils.register_keras_serializable()
class InputPtEtaPhiELayer(keras.layers.Layer):
    """
    Converts input features from (pt, eta, phi, E) to (px, py, pz, E) and optionally applies logarithmic scaling to pt and E.
    Optionally applies logarithmic scaling to pt and E.

    Args:
        log_variables (bool): Whether to apply logarithmic scaling to pt and E.
        padding_value (float): The value to use for padding masked entries.
    """

    def __init__(self, log_variables, padding_value, **kwargs):
        super().__init__(**kwargs)
        self.log_variables = log_variables
        self.padding_value = padding_value

    def call(self, inputs, mask=None):
        # Split input features
        pt = inputs[..., 0:1]
        eta = inputs[..., 1:2]
        phi = inputs[..., 2:3]
        energy = inputs[..., 3:4]
        residual = inputs[..., 4:]

        if mask is not None:
            # Ensure mask is broadcastable to feature dimension
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)

            # Replace masked entries with zeros for stable computation
            safe_pt = tf.where(mask, pt, tf.zeros_like(pt))
            safe_eta = tf.where(mask, eta, tf.zeros_like(eta))
            safe_phi = tf.where(mask, phi, tf.zeros_like(phi))
            safe_energy = tf.where(mask, energy, tf.ones_like(energy))
        else:
            # No mask passed — process everything
            safe_pt, safe_eta, safe_phi, safe_energy = pt, eta, phi, energy
        if self.log_variables:
            safe_non_zero_energy = tf.where(
                safe_energy > 0, safe_energy, tf.ones_like(safe_energy) * 1e-6
            )
            safe_non_zero_pt = tf.where(
                safe_pt > 0, safe_pt, tf.ones_like(safe_pt) * 1e-6
            )
            safe_log_energy = tf.math.log(safe_non_zero_energy)
            safe_log_pt = tf.math.log(safe_non_zero_pt)

        # Compute Cartesian components
        px = safe_pt * tf.cos(safe_phi)
        py = safe_pt * tf.sin(safe_phi)
        pz = safe_pt * tf.sinh(tf.clip_by_value(safe_eta, -10.0, 10.0))

        # Concatenate outputs
        if self.log_variables:
            outputs = tf.concat(
                [px, py, pz, safe_energy, safe_log_energy, safe_log_pt, residual],
                axis=-1,
            )
        else:
            outputs = tf.concat([px, py, pz, safe_energy, residual], axis=-1)

        # If mask exists, restore masked entries to padding value
        if mask is not None:
            outputs = tf.where(
                mask, outputs, tf.ones_like(outputs) * self.padding_value
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "log_variables": self.log_variables,
                "padding_value": self.padding_value,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class InputMetPhiLayer(keras.layers.Layer):
    """
    Converts MET representation from (met, met_phi)
    to (met_x, met_y).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        met = inputs[..., 0:1]
        met_phi = inputs[..., 1:2]

        met_x = met * tf.cos(met_phi)
        met_y = met * tf.sin(met_phi)

        return tf.concat([met_x / 1e3, met_y / 1e3], axis=-1)

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable()
class InputMetLayer(keras.layers.Layer):
    """
    Converts MET representation from (met, met_phi)
    to (met_x, met_y).
    """

    def __init__(self, log_variables=True, **kwargs):
        super().__init__(**kwargs)
        self.log_variables = log_variables

    def call(self, inputs):
        met = inputs[..., 0:1]
        met_phi = inputs[..., 1:2]

        log_met = tf.math.log(tf.where(met > 0, met, tf.ones_like(met) * 1e-6))
        met_cos_phi = tf.cos(met_phi)
        met_sin_phi = tf.sin(met_phi)
        output_tensors = [log_met, met_cos_phi, met_sin_phi]
        if self.log_variables:
            output_tensors.append(met)

        return tf.concat(output_tensors, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "log_variables": self.log_variables,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class InputNuFlowsPriors(keras.layers.Layer):
    """
    Converts (nu_px, nu_py, nu_pz, nubar_px, nubar_py, nubar_pz) to (((nu_pt, nu_eta, nu_phi ),(nubar_pt, nubar_eta, nubar_phi)))

    """

    def __init__(self, log_variables=True, **kwargs):
        self.log_variables = log_variables
        super().__init__(**kwargs)

    def call(self, inputs):
        nu_px = inputs[..., 0, 0:1]
        nu_py = inputs[..., 0, 1:2]
        nu_pz = inputs[..., 0, 2:3]
        nubar_px = inputs[..., 1, 0:1]
        nubar_py = inputs[..., 1, 1:2]
        nubar_pz = inputs[..., 1, 2:3]

        nu_pt = tf.sqrt(nu_px**2 + nu_py**2)
        nu_eta = tf.atanh(
            tf.clip_by_value(
                nu_pz / tf.sqrt(nu_px**2 + nu_py**2 + nu_pz**2 + 1e-8), -0.9999, 0.9999
            )
        )
        nu_phi = tf.atan2(nu_py, nu_px)

        nubar_pt = tf.sqrt(nubar_px**2 + nubar_py**2)
        nubar_eta = tf.atanh(
            tf.clip_by_value(
                nubar_pz / tf.sqrt(nubar_px**2 + nubar_py**2 + nubar_pz**2 + 1e-8),
                -0.9999,
                0.9999,
            )
        )
        nubar_phi = tf.atan2(nubar_py, nubar_px)
        if self.log_variables:
            nu_pt = tf.math.log(tf.where(nu_pt > 0, nu_pt, tf.ones_like(nu_pt) * 1e-6))
            nubar_pt = tf.math.log(
                tf.where(nubar_pt > 0, nubar_pt, tf.ones_like(nubar_pt) * 1e-6)
            )
        nu_vect = tf.concat([nu_pt, nu_eta, nu_phi, tf.ones_like(nu_pt) * 1], axis=-1)
        nubar_vect = tf.concat(
            [nubar_pt, nubar_eta, nubar_phi, tf.ones_like(nu_pt) * -1], axis=-1
        )
        return tf.stack([nu_vect, nubar_vect], axis=1)

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable()
class ProcessPtEtaPhiELayer(keras.layers.Layer):
    """
    Applies logarithmic scaling to pt and E.

    Args:
        log_variables (bool): Whether to apply logarithmic scaling to pt and E.
        padding_value (float): The value to use for padding masked entries.
    """

    def __init__(self, padding_value, log_variables=True, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value
        self.log_variables = log_variables

    def call(self, inputs, mask=None):
        # Split input features
        pt = inputs[..., 0:1]
        eta = inputs[..., 1:2]
        phi = inputs[..., 2:3]
        energy = inputs[..., 3:4]
        residual = inputs[..., 4:]

        if mask is not None:
            # Ensure mask is broadcastable to feature dimension
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)

            # Replace masked entries with zeros for stable computation
            safe_pt = tf.where(mask, pt, tf.zeros_like(pt))
            safe_eta = tf.where(mask, eta, tf.zeros_like(eta))
            safe_phi = tf.where(mask, phi, tf.zeros_like(phi))
            safe_energy = tf.where(mask, energy, tf.ones_like(energy))
        else:
            # No mask passed — process everything
            safe_pt, safe_eta, safe_phi, safe_energy = pt, eta, phi, energy

        safe_non_zero_energy = tf.where(
            safe_energy > 0, safe_energy, tf.ones_like(safe_energy) * 1e-6
        )
        safe_non_zero_pt = tf.where(safe_pt > 0, safe_pt, tf.ones_like(safe_pt) * 1e-6)
        safe_log_energy = tf.math.log(safe_non_zero_energy)
        safe_log_pt = tf.math.log(safe_non_zero_pt)
        safe_cos_phi = tf.cos(safe_phi)
        safe_sin_phi = tf.sin(safe_phi)

        output_tensors = [
            safe_log_pt,
            safe_eta,
            safe_cos_phi,
            safe_sin_phi,
            safe_log_energy,
        ]

        if self.log_variables:
            output_tensors.extend([safe_energy, safe_pt])

        output_tensors.append(residual)

        # Concatenate outputs
        outputs = tf.concat(
            output_tensors,
            axis=-1,
        )
        # If mask exists, restore masked entries to padding value
        if mask is not None:
            outputs = tf.where(
                mask, outputs, tf.ones_like(outputs) * self.padding_value
            )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding_value": self.padding_value,
                "log_variables": self.log_variables,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class ComputeHighLevelFeatures(keras.layers.Layer):
    """
    Computes high-level feature relational features between jets and leptons.
    """

    def __init__(self, padding_value=-999, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, jet_input, lepton_input, jet_mask=None, lepton_mask=None):
        # Expand dimensions for broadcasting

        jet_lepton_mask = None
        # Create combined mask - ensure it's always (B, J, L)
        if jet_mask is not None or lepton_mask is not None:
            if jet_mask is not None:
                jet_mask = tf.expand_dims(jet_mask, axis=2)  # (B, J, 1)
                jet_mask = tf.cast(jet_mask, tf.bool)
            if lepton_mask is not None:
                lepton_mask = tf.expand_dims(lepton_mask, axis=1)  # (B, 1, L)
                lepton_mask = tf.cast(lepton_mask, tf.bool)

            # Combine masks (broadcasting will create (B, J, L))
            if jet_mask is not None and lepton_mask is not None:
                jet_lepton_mask = jet_mask & lepton_mask  # (B, J, L)
            elif jet_mask is not None:
                jet_lepton_mask = jet_mask  # (B, J, 1)
            else:
                jet_lepton_mask = lepton_mask  # (B, 1, L)        # Split jet features
            jet_lepton_mask = tf.cast(jet_lepton_mask, tf.bool)  # (B, J, L, 1)
            jet_lepton_mask = tf.expand_dims(jet_lepton_mask, axis=-1)

        jet_px = tf.expand_dims(jet_input[..., 0:1], axis=2)  # Shape: (B, J, 1, 1)
        jet_py = tf.expand_dims(jet_input[..., 1:2], axis=2)
        jet_pz = tf.expand_dims(jet_input[..., 2:3], axis=2)
        jet_E = tf.expand_dims(jet_input[..., 3:4], axis=2)

        # Split lepton features
        lep_px = tf.expand_dims(lepton_input[..., 0:1], axis=1)  # Shape: (B, 1, L,1)
        lep_py = tf.expand_dims(lepton_input[..., 1:2], axis=1)
        lep_pz = tf.expand_dims(lepton_input[..., 2:3], axis=1)
        lep_E = tf.expand_dims(lepton_input[..., 3:4], axis=1)

        # Compute delta eta using arctanh for pseudorapidity
        jet_p_perp_sq = jet_px**2 + jet_py**2
        jet_p_sq = jet_p_perp_sq + jet_pz**2
        jet_eta = tf.atanh(
            tf.clip_by_value(jet_pz / tf.sqrt(jet_p_sq + 1e-8), -0.9999, 0.9999)
        )

        lep_p_perp_sq = lep_px**2 + lep_py**2
        lep_p_sq = lep_p_perp_sq + lep_pz**2
        lep_eta = tf.atanh(
            tf.clip_by_value(lep_pz / tf.sqrt(lep_p_sq + 1e-8), -0.9999, 0.9999)
        )

        delta_eta = jet_eta - lep_eta

        # Compute delta phi
        delta_phi = tf.atan2(jet_py, jet_px) - tf.atan2(lep_py, lep_px)
        delta_phi = tf.math.floormod(
            delta_phi + tf.constant(math.pi), 2 * tf.constant(math.pi)
        ) - tf.constant(math.pi)

        # Compute delta R
        delta_R = tf.sqrt(delta_eta**2 + delta_phi**2)

        # Compute invariant mass
        total_E = jet_E + lep_E
        total_px = jet_px + lep_px
        total_py = jet_py + lep_py
        total_pz = jet_pz + lep_pz

        invariant_mass = tf.sqrt(
            tf.maximum(total_E**2 - total_px**2 - total_py**2 - total_pz**2, 0.0)
        )

        # Concatenate high-level features
        HLF = tf.concat([delta_R, invariant_mass], axis=-1)

        # Apply masking if available
        if jet_lepton_mask is not None:
            HLF = tf.where(
                jet_lepton_mask,
                HLF,
                tf.fill(tf.shape(HLF), self.padding_value),
            )

        return HLF  # Shape: (batch_size, n_jets, n_leptons, 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding_value": self.padding_value,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class ComputeHighLevelFeatures_from_PtEtaPhiE(keras.layers.Layer):
    """
    Computes high-level feature relational features between jets and leptons.
    """

    def __init__(self, padding_value=-999, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, jet_input, lepton_input, jet_mask=None, lepton_mask=None):
        # Expand dimensions for broadcasting

        jet_lepton_mask = None
        # Create combined mask - ensure it's always (B, J, L)
        if jet_mask is not None or lepton_mask is not None:
            if jet_mask is not None:
                jet_mask = tf.expand_dims(jet_mask, axis=2)  # (B, J, 1)
                jet_mask = tf.cast(jet_mask, tf.bool)
            if lepton_mask is not None:
                lepton_mask = tf.expand_dims(lepton_mask, axis=1)  # (B, 1, L)
                lepton_mask = tf.cast(lepton_mask, tf.bool)

            # Combine masks (broadcasting will create (B, J, L))
            if jet_mask is not None and lepton_mask is not None:
                jet_lepton_mask = jet_mask & lepton_mask  # (B, J, L)
            elif jet_mask is not None:
                jet_lepton_mask = jet_mask  # (B, J, 1)
            else:
                jet_lepton_mask = lepton_mask  # (B, 1, L)        # Split jet features
            jet_lepton_mask = tf.cast(jet_lepton_mask, tf.bool)  # (B, J, L, 1)
            jet_lepton_mask = tf.expand_dims(jet_lepton_mask, axis=-1)

        jet_pt = tf.expand_dims(jet_input[..., 0:1], axis=2)  # Shape: (B, J, 1, 1)
        jet_eta = tf.expand_dims(jet_input[..., 1:2], axis=2)
        jet_phi = tf.expand_dims(jet_input[..., 2:3], axis=2)
        jet_E = tf.expand_dims(jet_input[..., 3:4], axis=2)

        # Split lepton features
        lep_pt = tf.expand_dims(lepton_input[..., 0:1], axis=1)  # Shape: (B, 1, L,1)
        lep_eta = tf.expand_dims(lepton_input[..., 1:2], axis=1)
        lep_phi = tf.expand_dims(lepton_input[..., 2:3], axis=1)
        lep_E = tf.expand_dims(lepton_input[..., 3:4], axis=1)

        jet_px = jet_pt * tf.cos(jet_phi)
        jet_py = jet_pt * tf.sin(jet_phi)
        jet_pz = jet_pt * tf.sinh(tf.clip_by_value(jet_eta, -10.0, 10.0))

        lep_px = lep_pt * tf.cos(lep_phi)
        lep_py = lep_pt * tf.sin(lep_phi)
        lep_pz = lep_pt * tf.sinh(tf.clip_by_value(lep_eta, -10.0, 10.0))

        # Compute delta phi
        delta_phi = jet_phi - lep_phi
        delta_phi = tf.math.floormod(
            delta_phi + tf.constant(math.pi), 2 * tf.constant(math.pi)
        ) - tf.constant(math.pi)
        delta_eta = jet_eta - lep_eta

        # Compute delta R
        delta_R = tf.sqrt(delta_eta**2 + delta_phi**2)

        # Compute invariant mass
        total_E = jet_E + lep_E
        total_px = jet_px + lep_px
        total_py = jet_py + lep_py
        total_pz = jet_pz + lep_pz

        invariant_mass = tf.sqrt(
            tf.maximum(total_E**2 - total_px**2 - total_py**2 - total_pz**2, 0.0)
        )

        # Concatenate high-level features
        HLF = tf.concat([delta_R, invariant_mass], axis=-1)

        # Apply masking if available
        if jet_lepton_mask is not None:
            HLF = tf.where(
                jet_lepton_mask,
                HLF,
                tf.fill(tf.shape(HLF), self.padding_value),
            )

        return HLF  # Shape: (batch_size, n_jets, n_leptons, 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding_value": self.padding_value,
            }
        )
        return config
