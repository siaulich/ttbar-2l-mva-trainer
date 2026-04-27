import tensorflow as tf
from keras import layers, Model
import keras as keras


@keras.utils.register_keras_serializable()
class SplitInputsLayer(layers.Layer):
    def __init__(self, input_shapes, **kwargs):
        super().__init__(**kwargs)
        self.input_shapes = [tuple(s) for s in input_shapes]
        # Compute flat sizes once in Python
        self.sizes = [int(tf.reduce_prod(s)) for s in self.input_shapes]

    def call(self, inputs):
        # Split flat tensor
        splits = tf.split(inputs, self.sizes, axis=-1)
        # Reshape into original forms
        reshaped = [
            tf.reshape(s, (-1, *shape)) for s, shape in zip(splits, self.input_shapes)
        ]
        return reshaped

    def get_config(self):
        config = super().get_config()
        config.update({"input_shapes": self.input_shapes})
        return config
