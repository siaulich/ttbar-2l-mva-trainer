from .configs import DataConfig, LoadConfig, load_yaml_config, get_load_config_from_yaml
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    import tensorflow as tf

    print("Could not set TensorFlow GPU memory growth.")
    pass

import warnings

warnings.filterwarnings("ignore")

import logging

logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
