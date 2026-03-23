import numpy as np


def lorentz_vector_from_pt_eta_phi_e(pt, eta, phi, e, padding_value=-999):
    """
    Computes the four-momentum vector components from pt, eta, phi, and energy.

    Args:
        pt (np.ndarray): Transverse momentum.
        eta (np.ndarray): Pseudorapidity.
        phi (np.ndarray): Azimuthal angle.
        e (np.ndarray): Energy.
    Returns:
        tuple: A tuple containing the four-momentum components (px, py, pz, e).
    """
    mask = (
        (pt == padding_value)
        | (eta == padding_value)
        | (phi == padding_value)
        | (e == padding_value)
    )
    phi = np.where(mask, 0, phi)
    eta = np.where(mask, 0, eta)
    pt = np.where(mask, 0, pt)
    e = np.where(mask, padding_value, e)
    px = np.where(mask, padding_value, pt * np.cos(phi))
    py = np.where(mask, padding_value, pt * np.sin(phi))
    pz = np.where(mask, padding_value, pt * np.sinh(eta))
    return px, py, pz, e


def lorentz_vector_array_from_pt_eta_phi_e(pt, eta, phi, e, padding_value=-999):
    """
    Computes the four-momentum vector components from pt, eta, phi, and energy.

    Args:
        pt (np.ndarray): Transverse momentum.
        eta (np.ndarray): Pseudorapidity.
        phi (np.ndarray): Azimuthal angle.
        e (np.ndarray): Energy.
    Returns:
        tuple: A tuple containing the four-momentum components (px, py, pz, e).
    """
    mask = (
        (pt == padding_value)
        | (eta == padding_value)
        | (phi == padding_value)
        | (e == padding_value)
    )
    phi = np.where(mask, 0, phi)
    eta = np.where(mask, 0, eta)
    pt = np.where(mask, 0, pt)
    e = np.where(mask, padding_value, e)
    px = np.where(mask, padding_value, pt * np.cos(phi))
    py = np.where(mask, padding_value, pt * np.sin(phi))
    pz = np.where(mask, padding_value, pt * np.sinh(eta))
    return np.stack((px, py, pz, e), axis=-1)


def compute_mass_from_lorentz_vector(px, py, pz, e, padding_value=-999):
    """
    Computes the invariant mass from four-momentum components.

    Args:
        px (np.ndarray): x-component of momentum.
        py (np.ndarray): y-component of momentum.
        pz (np.ndarray): z-component of momentum.
        e (np.ndarray): Energy.
    Returns:
        np.ndarray: The invariant mass.
    """
    # Create mask for padding and invalid values (NaN/inf)
    mask = (
        (px == padding_value)
        | (py == padding_value)
        | (pz == padding_value)
        | (e == padding_value)
    )
    mask = mask | np.isnan(px) | np.isnan(py) | np.isnan(pz) | np.isnan(e)
    mask = mask | np.isinf(px) | np.isinf(py) | np.isinf(pz) | np.isinf(e)

    px = np.where(mask, 0, px)
    py = np.where(mask, 0, py)
    pz = np.where(mask, 0, pz)
    e = np.where(mask, 0, e)

    mass_squared = np.where(mask, padding_value, e**2 - (px**2 + py**2 + pz**2))
    mass_squared = np.maximum(
        mass_squared, 0
    )  # Prevent negative values due to numerical errors
    return np.sqrt(mass_squared)


def compute_mass_from_lorentz_vector_array(array, padding_value=-999):
    """
    Computes the invariant mass from an array of four-momentum components.

    Args:
        array (np.ndarray): An array with shape (..., 4) containing px, py, pz, and e.
    Returns:
        array (np.ndarray): An array with shape (...) containing the invariant mass.
    """
    px = array[..., 0]
    py = array[..., 1]
    pz = array[..., 2]
    e = array[..., 3]
    return compute_mass_from_lorentz_vector(px, py, pz, e, padding_value)


def lorentz_vector_from_PtEtaPhiE_array(array, padding_value=-999):
    """
    Computes the four-momentum vector components from an array of pt, eta, phi, and energy.

    Args:
        array (np.ndarray): An array with shape (..., 4) containing pt, eta, phi, and energy.
    Returns:
        array (np.ndarray): An array with shape (..., 4) containing the four-momentum components (px, py, pz, e).
    """
    pt = array[..., 0]
    eta = array[..., 1]
    phi = array[..., 2]
    e = array[..., 3]
    px, py, pz, e = lorentz_vector_from_pt_eta_phi_e(pt, eta, phi, e, padding_value)
    return np.stack((px, py, pz, e), axis=-1)


def lorentz_vector_from_neutrino_momenta_array(array, padding_value=-999):
    """
    Computes the four-momentum vector components for neutrinos from an array of (px, py, pz).

    Args:
        array (np.ndarray): An array with shape (..., 3) containing px, py, and pz.
    Returns:
        array (np.ndarray): An array with shape (..., 4) containing the four-momentum components (px, py, pz, e).
    """
    px = array[..., 0]
    py = array[..., 1]
    pz = array[..., 2]
    e = np.where(
        (px == padding_value) | (py == padding_value) | (pz == padding_value),
        padding_value,
        np.sqrt(px**2 + py**2 + pz**2),
    )
    return np.stack((px, py, pz, e), axis=-1)


def compute_pt_from_lorentz_vector_array(array, padding_value=-999):
    """
    Computes the transverse momentum (pt) from an array of four-momentum components.

    Args:
        array (np.ndarray): An array with shape (..., 4) containing px, py, pz, and e.
    Returns:
        array (np.ndarray): An array with shape (...) containing the transverse momentum (pt).
    """
    px = array[..., 0]
    py = array[..., 1]
    mask = (px == padding_value) | (py == padding_value)
    px = np.where(mask, 0, px)
    py = np.where(mask, 0, py)
    pt = np.sqrt(px**2 + py**2)
    return np.where(mask, padding_value, pt)


def project_vectors_onto_axis(
    vectors: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """
    Project vectors onto a given axis.

    Args:
        vectors: Array of vectors (n_events, 3)
        axis: Axis to project onto (3,)

    Returns:
        Array of projected components (n_events,)
    """
    axis_norm = np.linalg.norm(axis)
    axis_norm = np.clip(axis_norm, 1e-10, None)  # Avoid division by zero

    # Check for finite values
    valid_axis = np.isfinite(axis_norm) & np.all(np.isfinite(axis))
    axis_norm = np.where(valid_axis, axis_norm, 1.0)

    unit_axis = axis / axis_norm
    unit_axis = np.where(valid_axis, unit_axis, 0.0)

    projections = np.sum(vectors * unit_axis, axis=-1)

    # Return 0 for invalid cases
    projections = np.where(
        np.isfinite(projections) & ~np.isnan(projections), projections, 0.0
    )
    return projections


def angle_vectors(a: np.ndarray, b: np.ndarray, axis=-1) -> np.ndarray:
    unit_a = a / np.linalg.norm(a, axis=axis)[..., np.newaxis]
    unit_b = b / np.linalg.norm(b, axis=axis)[..., np.newaxis]
    return np.arccos(np.clip(np.sum(unit_a * unit_b, axis=axis), -1.0, 1.0))



def PtEtaPhi_to_vector3(vector: np.ndarray) -> np.ndarray:
    pt, eta, phi = vector[..., 0], vector[..., 1], vector[..., 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return np.stack((px, py, pz), axis=-1)


def vector3_to_PtEtaPhi(vector: np.ndarray) -> np.ndarray:
    px, py, pz = vector[..., 0], vector[..., 1], vector[..., 2]
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / np.clip(pt, 1e-10, None))
    phi = np.arctan2(py, px)
    return np.stack((pt, eta, phi), axis=-1)

def cos_angle_vectors(vec1: np.ndarray, vec2: np.ndarray, axis=-1) -> np.ndarray:
    """
    Computes the cosine of the angle between two vectors.

    Args:
        a: First vector (n_events, 3)
        b: Second vector (n_events, 3)
        axis: Axis along which to compute the angle (default: -1)

    Returns:
        Array of cosine values (n_events,)
    """
    a = vec1[..., :3]  # Ensure we only use the spatial components
    b = vec2[..., :3]

    dot_product = np.sum(a * b, axis=axis)
    norm_a = np.linalg.norm(a, axis=axis)
    norm_b = np.linalg.norm(b, axis=axis)

    # Avoid division by zero and invalid values
    valid = (norm_a > 1e-10) & (norm_b > 1e-10) & np.isfinite(dot_product)
    cos_angle = np.where(valid, dot_product / (norm_a * norm_b), 0.0)
    return np.clip(cos_angle, -1.0, 1.0)

def magnitude_of_vector(vec: np.ndarray, axis=-1) -> np.ndarray:
    """
    Computes the magnitude of a vector.

    Args:
        vector: Input vector (n_events, 3)
        axis: Axis along which to compute the magnitude (default: -1)

    Returns:
        Array of magnitudes (n_events,)
    """
    a = vec[..., :3]  # Ensure we only use the spatial components
    return np.linalg.norm(a, axis=axis)