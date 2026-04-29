from ..base_classes.reconstruction_base import EventReconstructorBase
from .keras_ff_reco_base import KerasFFRecoBase
import numpy as np
from ..preprocessing.training_data_loader import DataConfig
from typing import Union


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


class PerfectAssignmentReconstructor(KerasFFRecoBase, GroundTruthReconstructor):
    def __init__(
        self,
        config: DataConfig,
        neutrino_reco_name="ML Neutrino Reco",
        assignment_name="True Assignment",
        load_model_path=None,
    ):
        KerasFFRecoBase.__init__(
            self,
            name=neutrino_reco_name,
            config=config,
            load_model_path=load_model_path,
        )
        EventReconstructorBase.__init__(
            self,
            config=config,
            assignment_name=assignment_name,
            full_reco_name=assignment_name
            + (" + " if assignment_name != "" else "")
            + neutrino_reco_name,
            neutrino_name=neutrino_reco_name,
            perform_regression=True,
            use_nu_flows=False,
        )
        self.config = config

    def predict_indices(self, data_dict):
        return GroundTruthReconstructor.predict_indices(self, data_dict)

    def reconstruct_neutrinos(self, data_dict):
        return KerasFFRecoBase.reconstruct_neutrinos(self, data_dict)

    def complete_forward_pass(self, data):
        model_prediction = KerasFFRecoBase.complete_forward_pass(self, data)
        assignment = GroundTruthReconstructor.predict_indices(self, data)
        return assignment, model_prediction[1]


class CompositeNeutrinoComponentReconstructor(KerasFFRecoBase):
    def __init__(
        self,
        config: DataConfig,
        use_nu_flows=False,
        axis: Union[int, tuple] = -1,
        true_assignment=False,
        name="ML Model",
    ):
        EventReconstructorBase.__init__(
            self,
            config=config,
            assignment_name="Ground truth" if true_assignment else name,
            neutrino_name=(r"$\nu^2$-Flows" if use_nu_flows else r"True $\nu$")
            + "("
            + (
                ["x", "y", "z"][axis]
                if isinstance(axis, int)
                else (["x", "y", "z"][axis_i] for axis_i in axis).join(",")
            )
            + ")",
            full_reco_name=(name)
            + " + "
            + (r"$\nu^2$-Flows" if use_nu_flows else r"True $\nu$")
            + "("
            + (
                ["x", "y", "z"][axis]
                if isinstance(axis, int)
                else (["x", "y", "z"][axis_i] for axis_i in axis).join(",")
            )
            + ")",
            perform_regression=True,
        )
        self.ground_truth_assigner = GroundTruthReconstructor(
            config=config,
            use_nu_flows=use_nu_flows,
        )
        self.config = config
        self.axis = axis
        self.true_assignment = true_assignment

    def predict_indices(self, data_dict):
        if self.true_assignment:
            return self.ground_truth_assigner.predict_indices(self, data_dict)
        else:
            return KerasFFRecoBase.predict_indices(self, data_dict)

    def reconstruct_neutrinos(self, data_dict):
        ml_neutrino_reco = KerasFFRecoBase.reconstruct_neutrinos(self, data_dict)
        true_neutrino_reco = self.ground_truth_assigner.reconstruct_neutrinos(
            self, data_dict
        )
        ml_neutrino_reco[:, :, self.axis] = true_neutrino_reco[:, :, self.axis]
        return ml_neutrino_reco

    def complete_forward_pass(self, data):
        assignment, ml_neutrino_reco = super().complete_forward_pass(data)
        if self.true_assignment:
            assignment = self.ground_truth_assigner.predict_indices(data)
        true_neutrino_reco = self.ground_truth_assigner.reconstruct_neutrinos(data)
        ml_neutrino_reco[:, :, self.axis] = true_neutrino_reco[:, :, self.axis]

        return assignment, ml_neutrino_reco
