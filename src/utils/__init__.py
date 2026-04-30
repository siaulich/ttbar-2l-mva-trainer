from .losses import *
from .metrics import *
from .sample_weight import *
from .four_vector_arithmetics import *
from .plotting import *
from .deprecated import *
from keras.losses import Loss
from keras.metrics import Metric

def _get_loss(loss_name):
    if loss_name not in globals():
        raise ValueError(f"Loss '{loss_name}' not found in source.losses.")
    return globals()[loss_name]()


def _get_metric(metric_name):
    if metric_name not in globals():
        raise ValueError(f"Metric '{metric_name}' not found in source.metrics.")
    return globals()[metric_name]()


def get_custom_objects():
    custom_objects = {}
    for name, obj in globals().items():
        if isinstance(obj, type) and (
            issubclass(obj, Metric) or issubclass(obj, Loss)
        ):
            custom_objects[name] = obj
    return custom_objects
