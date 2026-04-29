from abc import ABC, abstractmethod
from ..configs import DataConfig


class BaseUtilityModel(ABC):
    def __init__(
        self,
        config: DataConfig,
        assignment_name="BaseAssignmentUtility",
        full_reco_name="BaseFullRecoUtility",
        neutrino_name=None,
    ):
        self.config = config
        self.assignment_name = assignment_name
        self.full_reco_name = full_reco_name
        self.neutrino_name = full_reco_name if neutrino_name is None else neutrino_name

    def get_assignment_name(self):
        return self.assignment_name

    def get_full_reco_name(self):
        return self.full_reco_name

    def get_neutrino_name(self):
        return self.neutrino_name
