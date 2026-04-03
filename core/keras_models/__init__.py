from .assignment_RNN import *
from .assignment_transformer import *
from .regression_transformer import *
from .regression_transformer_variance import *
from .binned_regression_transformer import *
from ..reconstruction.baseline_methods import *
from ..base_classes import KerasMLWrapper

def _get_model(model_name) -> type[KerasMLWrapper]:
    if model_name not in globals():
        raise ValueError(f"Model '{model_name}' not found in keras_models.")
    if not issubclass(globals()[model_name], KerasMLWrapper):
        raise TypeError(f"Model '{model_name}' is not a subclass of KerasFFRecoBase.")
    return globals()[model_name]