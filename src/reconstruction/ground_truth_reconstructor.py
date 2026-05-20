from ..base_classes.reconstruction_base import EventReconstructorBase
from ..configs import DataConfig


class GroundTruthReconstructor(EventReconstructorBase):
    def __init__(
        self,
        config: DataConfig,
        neutrino_reco=None,
        assignment_name="True Assignment",
    ):
        super().__init__(
            config=config,
            assignment_name=assignment_name,
            full_reco_name=assignment_name
            + (
                config.neutrino_regression_method_labels.get(
                    neutrino_reco, neutrino_reco
                )
                if neutrino_reco
                else ""
            ),
            neutrino_name=(
                config.neutrino_regression_method_labels.get(
                    neutrino_reco, neutrino_reco
                )
                if neutrino_reco
                else None
            ),
            perform_regression=False,
            neutrino_reco=neutrino_reco,
        )
        self.config = config

    def predict_indices(self, data_dict):
        return data_dict["assignment"]

    def reconstruct_neutrinos(self, data_dict):
        return super().reconstruct_neutrinos(data_dict)
