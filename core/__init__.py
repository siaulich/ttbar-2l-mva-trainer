from .configs import DataConfig, LoadConfig, load_yaml_config, get_load_config_from_yaml

try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    import tensorflow as tf

    print("Could not set TensorFlow GPU memory growth.")
    pass
