from ..base_classes.reconstruction_base import EventReconstructorBase
from ..configs import DataConfig


class GroundTruthReconstructor(EventReconstructorBase):
    def __init__(
        self,
        config: DataConfig,
        use_nu_flows=False,
        assignment_name="True Assignment",
    ):
        super().__init__(
            config=config,
            assignment_name=assignment_name,
            full_reco_name=assignment_name
            + (r" + " if assignment_name != "" else r"")
            + (r"$\nu^2$-Flows" if use_nu_flows else r"True $\nu$"),
            neutrino_name=r"$\nu^2$-Flows" if use_nu_flows else r"True $\nu$",
            perform_regression=False,
            use_nu_flows=use_nu_flows,
        )
        self.config = config

    def predict_indices(self, data_dict):
        return data_dict["assignment"]

    def reconstruct_neutrinos(self, data_dict):
        return super().reconstruct_neutrinos(data_dict)

