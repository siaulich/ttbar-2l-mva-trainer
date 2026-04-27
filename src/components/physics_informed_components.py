import keras as keras
import tensorflow as tf


def pt_eta_phi_e_tensor_to_4_vect_tensor(tensor: tf.Tensor):
    """Convert (pt, eta, phi, E) to (px, py, pz, E) with numerical stability"""
    pt = tensor[..., 0:1]
    eta = tensor[..., 1:2]
    phi = tensor[..., 2:3]
    energy = tensor[..., 3:4]

    # Clip eta to reasonable range to avoid sinh explosion
    eta = tf.clip_by_value(eta, -5.0, 5.0)

    # Ensure positive pt
    pt = tf.maximum(pt, 1e-8)

    px = pt * tf.cos(phi)
    py = pt * tf.sin(phi)
    pz = pt * tf.sinh(eta)

    four_vector = tf.concat([px, py, pz, energy], axis=-1)
    return four_vector


def neutrino_3_vect_to_4_vect_tensor(tensor: tf.Tensor):
    """Convert (px, py, pz) to (px, py, pz, E) assuming massless"""
    px = tensor[..., 0:1]
    py = tensor[..., 1:2]
    pz = tensor[..., 2:3]

    momentum_squared = px**2 + py**2 + pz**2
    energy = tf.sqrt(momentum_squared + 1e-8)

    four_vector = tf.concat([px, py, pz, energy], axis=-1)
    return four_vector


def reco_W_mass(neutrino_momenta, lepton_momenta):

    # Convert to 4-vectors
    lepton_momenta_4v = pt_eta_phi_e_tensor_to_4_vect_tensor(
        lepton_momenta
    )  # (batch_size, 2, 4)
    neutrino_momenta_4v = neutrino_3_vect_to_4_vect_tensor(
        neutrino_momenta
    )  # (batch_size, 2, 4)

    # Sum momenta
    total_momenta = lepton_momenta_4v + neutrino_momenta_4v

    total_energy = total_momenta[..., 3]
    total_momentum_squared = tf.reduce_sum(total_momenta[..., 0:3] ** 2, axis=-1)
    # Invariant mass with numerical stability
    invariant_mass_squared = total_energy**2 - total_momentum_squared
    # Ensure non-negative before sqrt
    invariant_mass_squared = tf.maximum(invariant_mass_squared, 0.0)
    invariant_mass = tf.sqrt(invariant_mass_squared + 1e-6) / 1e3  # Convert to GeV

    return invariant_mass


def reco_W_mass_deviation(neutrino_momenta, lepton_momenta):
    W_MASS = 80.379  # GeV
    invariant_mass = reco_W_mass(neutrino_momenta, lepton_momenta)

    # Normalized mass difference
    mass_diff = (invariant_mass - W_MASS) / W_MASS
    mass_diff_square = tf.square(mass_diff)
    mass_loss = mass_diff_square
    # Safety check
    mass_loss = tf.where(
        tf.math.is_finite(mass_loss),
        mass_loss,
        tf.constant(0.0, dtype=mass_loss.dtype),
    )

    mass_loss = tf.reduce_mean(mass_loss, axis=-1)

    return mass_loss


@keras.utils.register_keras_serializable()
class PhysicsInformedLoss(keras.layers.Layer):
    def __init__(self, name="reco_mass_deviation", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, neutrino_momenta, lepton_momenta):
        mass_loss = reco_W_mass_deviation(neutrino_momenta, lepton_momenta)
        return mass_loss

    def get_config(self):
        config = super().get_config()
        return config
