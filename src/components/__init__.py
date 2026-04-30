from .attention_blocks import *
from .mlp import *
from .masking import *
from .onnx_support import *
from .input_layers import *
from .output_layers import *
from .regression_components import *
from .physics_informed_components import *
from .utils import *
from keras.layers import Layer


def get_custom_layers():
    custom_layers = {}
    for name, obj in globals().items():
        if isinstance(obj, type) and issubclass(obj, Layer):
            custom_layers[name] = obj
    return custom_layers
