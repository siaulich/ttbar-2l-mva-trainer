"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from .physics_calculations import (
    ResolutionCalculator,
    lorentz_vector_from_PtEtaPhiE_array,
    select_jets,
    boost,
    c_hel,
    c_han,
)
from core.utils import (
    compute_pt_from_lorentz_vector_array,
    compute_eta_from_lorentz_vector_array,
    compute_phi_from_lorentz_vector_array,
    lorentz_vector_from_neutrino_momenta_array,
    compute_mass_from_lorentz_vector_array,
    project_vectors_onto_axis,
    angle_vectors,
    cos_angle_vectors,
    delta_phi_top_lepton_helicity,
)

# Function aliases
make_4vect = lorentz_vector_from_PtEtaPhiE_array
make_nu_4vect = lorentz_vector_from_neutrino_momenta_array


reconstruction_variable_configs = {
    "ttbar_mass": {
        "compute_func": lambda l, j, n: compute_mass_from_lorentz_vector_array(
            l[:, 0, :4]
            + j[:, 0, :4]
            + n[:, 0, :]
            + l[:, 1, :4]
            + j[:, 1, :4]
            + n[:, 1, :]
        )
        / 1e3,
        "extract_func": lambda X: compute_mass_from_lorentz_vector_array(
            make_4vect(X["top_truth"][:, 0, :4]) + make_4vect(X["top_truth"][:, 1, :4])
        )
        / 1e3,
        "label": r"$m(t\bar{t})$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (340, 1000),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t\bar{t})$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t\bar{t})$ Deviation",
        },
    },
    "c_han": {
        "compute_func": lambda l, j, n: c_han(
            l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :],
            l[:, 1, :4] + j[:, 1, :4] + n[:, 1, :],
            l[:, 0, :4],
            l[:, 1, :4],
        ),
        "extract_func": lambda X: c_han(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
            (X["lepton_truth"][:, 0, :4]),
            (X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{han}}$",
        "use_relative_deviation": False,
        "xlims": (-1, 1),
        "bins": 10,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{han}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{han}}$ Deviation",
        },
    },
    "c_hel": {
        "compute_func": lambda l, j, n: c_hel(
            l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :],
            l[:, 1, :4] + j[:, 1, :4] + n[:, 1, :],
            l[:, 0, :4],
            l[:, 1, :4],
        ),
        "extract_func": lambda X: c_hel(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
            (X["lepton_truth"][:, 0, :4]),
            (X["lepton_truth"][:, 1, :4]),
        ),
        "label": r"$c_{\text{hel}}$",
        "use_relative_deviation": False,
        "xlims": (-1, 1),
        "bins": 10,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$c_{\text{hel}}$ Resolution",
            "ylabel_deviation": r"Mean $c_{\text{hel}}$ Deviation",
        },
    },
    "nu_mag": {
        "compute_func": lambda l, j, n: (np.linalg.norm(n[:, 0, :3], axis=-1) / 1e3,),
        "extract_func": lambda X: (
            np.linalg.norm(X["regression"][:, 0, :3], axis=-1) / 1e3,
        ),
        "label": r"$|\vec{p}(\nu)|$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\nu)|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\nu)|$ Deviation",
        },
    },
    "nubar_mag": {
        "compute_func": lambda l, j, n: (np.linalg.norm(n[:, 1, :3], axis=-1) / 1e3,),
        "extract_func": lambda X: (
            np.linalg.norm(X["regression"][:, 1, :3], axis=-1) / 1e3,
        ),
        "label": r"$|\vec{p}(\bar{\nu})|$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\bar{\nu})|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\bar{\nu})|$ Deviation",
        },
    },
    "nu_px": {
        "compute_func": lambda l, j, n: (n[:, 0, 0] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 0] / 1e3),
        "label": r"$p_{x}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_py": {
        "compute_func": lambda l, j, n: (n[:, 0, 1] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 1] / 1e3),
        "label": r"$p_{y}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(\nu)$ Deviation [GeV]",
        },
    },
    "nu_pz": {
        "compute_func": lambda l, j, n: (n[:, 0, 2] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 0, 2] / 1e3),
        "label": r"$p_{z}(\nu)$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(\nu)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(\nu)$ Deviation [GeV]",
        },
    },
    "nubar_px": {
        "compute_func": lambda l, j, n: (n[:, 1, 0] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 0] / 1e3),
        "label": r"$p_{x}(\bar{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{x}(\bar{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{x}(\bar{\nu})$ Deviation [GeV]",
        },
    },
    "nubar_py": {
        "compute_func": lambda l, j, n: (n[:, 1, 1] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 1] / 1e3),
        "label": r"$p_{y}(\bar{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{y}(\bar{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{y}(\bar{\nu})$ Deviation [GeV]",
        },
    },
    "nubar_pz": {
        "compute_func": lambda l, j, n: (n[:, 1, 2] / 1e3),
        "extract_func": lambda X: (X["regression"][:, 1, 2] / 1e3),
        "label": r"$p_{z}(\bar{\nu})$ [GeV]",
        "use_relative_deviation": False,
        "xlims": (-300, 300),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$p_{z}(\bar{\nu})$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $p_{z}(\bar{\nu})$ Deviation [GeV]",
        },
    },
    "nunubar_mag": {
        "compute_func": lambda l, j, n: (
            np.linalg.norm(n[:, 0, :3] + n[:, 1, :3], axis=-1) / 1e3,
        ),
        "extract_func": lambda X: (
            np.linalg.norm(
                X["regression"][:, 0, :3] + X["regression"][:, 1, :3], axis=-1
            )
            / 1e3,
        ),
        "label": r"$|\vec{p}(\nu) + \vec{p}(\bar{\nu})|$ [GeV]",
        "use_relative_deviation": True,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $|\vec{p}(\nu) + \vec{p}(\bar{\nu})|$ Resolution",
            "ylabel_deviation": r"Mean Relative $|\vec{p}(\nu) + \vec{p}(\bar{\nu})|$ Deviation",
        },
    },
    "cos_angle_nu_lep": {
        "compute_func": lambda l, j, n: cos_angle_vectors(n[:, 0, :], l[:, 0, :4]),
        "extract_func": lambda X: cos_angle_vectors(
            X["regression"][:, 0, :], X["lepton_truth"][:, 0, :4]
        ),
        "label": r"$\cos\theta(\nu, \ell)$",
        "use_relative_deviation": False,
        "xlims": (-1, 1),
        "bins": 10,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\cos\theta(\nu, \ell)$ Resolution",
            "ylabel_deviation": r"Mean $\cos\theta(\nu, \ell)$ Deviation",
        },
    },
    "cos_angle_nu_nubar": {
        "compute_func": lambda l, j, n: cos_angle_vectors(n[:, 0, :], n[:, 1, :]),
        "extract_func": lambda X: cos_angle_vectors(
            X["regression"][:, 0, :], X["regression"][:, 1, :]
        ),
        "label": r"$\cos\theta(\nu, \bar{\nu})$",
        "use_relative_deviation": False,
        "xlims": (-1, 1),
        "bins": 10,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\cos\theta(\nu, \bar{\nu})$ Resolution",
            "ylabel_deviation": r"Mean $\cos\theta(\nu, \bar{\nu})$ Deviation",
        },
    },
    "cos_angle_nunubar_bbarll": {
        "compute_func": lambda l, j, n: cos_angle_vectors(
            n[:, 0, :] + n[:, 1, :],
            l[:, 0, :4] + l[:, 1, :4] + j[:, 0, :4] + j[:, 1, :4],
        ),
        "extract_func": lambda X: cos_angle_vectors(
            X["regression"][:, 0, :] + X["regression"][:, 1, :],
            X["lepton_truth"][:, 0, :4]
            + X["lepton_truth"][:, 1, :4]
            + select_jets(make_4vect(X["jet_inputs"]), X["assignment"])[:, 0, :4]
            + select_jets(make_4vect(X["jet_inputs"]), X["assignment"])[:, 1, :4],
        ),
        "label": r"$\cos\theta(\nu\bar{\nu}, b\bar{b}\ell^+\ell^-)$",
        "use_relative_deviation": False,
        "xlims": (-1, 1),
        "bins": 10,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\cos\theta(\nu\bar{\nu}, b\bar{b}\ell^+\ell^-)$ Resolution",
            "ylabel_deviation": r"Mean $\cos\theta(\nu\bar{\nu}, b\bar{b}\ell^+\ell^-)$ Deviation",
        },
    },
    "top_energy": {
        "compute_func": lambda l, j, n: ((l[:, 0, 3] + j[:, 0, 3] + n[:, 0, 3]) / 1e3),
        "extract_func": lambda X: (make_4vect(X["top_truth"][:, 0, :4])[..., 3] / 1e3,),
        "label": r"$E(t)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (0, 500),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $E(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $E(t)$ Deviation",
        },
    },
    "top_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array(l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :])
            / 1e3
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array(make_4vect(X["top_truth"][:, 0, :4]))
            / 1e3
        ),
        "label": r"$p_{T}(t)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (0, 500),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(t)$ Deviation",
        },
    },
    "top_gamma": {
        "compute_func": lambda l, j, n: (
            (l[:, 0, 3] + j[:, 0, 3] + n[:, 0, 3])
            / compute_mass_from_lorentz_vector_array(
                l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :]
            )
        ),
        "extract_func": lambda X: (
            make_4vect(X["top_truth"][:, 0, :4])[..., 3]
            / compute_mass_from_lorentz_vector_array(
                make_4vect(X["top_truth"][:, 0, :4])
            )
        ),
        "label": r"$\gamma(t)$",
        "use_relative_deviation": False,
        "xlims": (0, 5),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\gamma(t)$ Resolution",
            "ylabel_deviation": r"Mean $\gamma(t)$ Deviation",
        },
    },
    "top_mass": {
        "compute_func": lambda l, j, n: compute_mass_from_lorentz_vector_array(
            l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :]
        )
        / 1e3,
        "extract_func": lambda X: compute_mass_from_lorentz_vector_array(
            make_4vect(X["top_truth"][:, 0, :4])
        )
        / 1e3,
        "label": r"$m(t)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (100, 300),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $m(t)$ Resolution",
            "ylabel_deviation": r"Mean Relative $m(t)$ Deviation",
        },
    },
    "W_mass": {
        "compute_func": lambda l, j, n: (
            compute_mass_from_lorentz_vector_array((l[:, 0, :4]) + (n[:, 0, :])) / 1e3
        ),
        "extract_func": lambda X: (
            compute_mass_from_lorentz_vector_array(
                X["lepton_truth"][:, 0, :4] + make_nu_4vect(X["regression"][:, 0, :])
            )
            / 1e3
        ),
        "label": r"$m(W)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (50, 110),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$m(W)$ Resolution [GeV]",
            "ylabel_deviation": r"Mean $m{x}(W)$ Deviation [GeV]",
        },
    },
    "W_energy": {
        "compute_func": lambda l, j, n: (
            ((l[:, 0, :4])[..., 3] + (n[:, 0, :])[..., 3]) / 1e3
        ),
        "extract_func": lambda X: (
            (
                X["lepton_truth"][:, 0, :4][..., 3]
                + make_nu_4vect(X["regression"][:, 0, :])[..., 3]
            )
            / 1e3
        ),
        "label": r"$E(W)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (0, 300),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $E(W)$ Resolution",
            "ylabel_deviation": r"Mean Relative $E(W)$ Deviation",
        },
    },
    "W_pt": {
        "compute_func": lambda l, j, n: (
            compute_pt_from_lorentz_vector_array((l[:, 0, :4]) + (n[:, 0, :])) / 1e3
        ),
        "extract_func": lambda X: (
            compute_pt_from_lorentz_vector_array(
                X["lepton_truth"][:, 0, :4] + make_nu_4vect(X["regression"][:, 0, :])
            )
            / 1e3
        ),
        "label": r"$p_{T}(W)$ [GeV]",
        "use_relative_deviation": True,
        "xlims": (0, 300),
        "bins": 30,
        "resolution": {
            "use_relative_deviation": True,
            "ylabel_resolution": r"Relative $p_{T}(W)$ Resolution",
            "ylabel_deviation": r"Mean Relative $p_{T}(W)$ Deviation",
        },
    },
    "delta_phi_ell_top": {
        "compute_func": lambda l, j, n: compute_phi_from_lorentz_vector_array(
            l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :]
        )
        - compute_phi_from_lorentz_vector_array(l[:, 0, :4]),
        "extract_func": lambda X: compute_phi_from_lorentz_vector_array(
            make_4vect(X["top_truth"][:, 0, :4])
        )
        - compute_phi_from_lorentz_vector_array(X["lepton_truth"][:, 0, :4]),
        "label": r"$\Delta\phi(\ell, t)$ [rad]",
        "use_relative_deviation": False,
        "xlims": (-np.pi, np.pi),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\Delta\phi(\ell, t)$ Resolution [rad]",
            "ylabel_deviation": r"Mean $\Delta\phi(\ell, t)$ De viation [rad]",
        },
    },
    "delta_phi_ell_top_helicity": {
        "compute_func": lambda l, j, n: delta_phi_top_lepton_helicity(
            l[:, 0, :4] + j[:, 0, :4] + n[:, 0, :],
            l[:, 1, :4] + j[:, 1, :4] + n[:, 1, :],
            l[:, 0, :4],
        ),
        "extract_func": lambda X: delta_phi_top_lepton_helicity(
            make_4vect(X["top_truth"][:, 0, :4]),
            make_4vect(X["top_truth"][:, 1, :4]),
            X["lepton_truth"][:, 0, :4],
        ),
        "label": r"$\Delta\phi(\ell_{hel}, t)$ [rad]",
        "use_relative_deviation": False,
        "xlims": (-np.pi, np.pi),
        "bins": 20,
        "resolution": {
            "use_relative_deviation": False,
            "ylabel_resolution": r"$\Delta\phi_{\text{helicity}}(\ell, t)$ Resolution [rad]",
            "ylabel_deviation": r"Mean $\Delta\phi_{\text{helicity}}(\ell, t)$ Deviation [rad]",
        },
    },
}
