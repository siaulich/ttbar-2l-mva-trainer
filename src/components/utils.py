import keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class TransposeLayer(keras.layers.Layer):
    def __init__(self, perm=None, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)

    def get_config(self):
        config = super().get_config()
        config.update({"perm": self.perm})
        return config


@keras.utils.register_keras_serializable()
class StopGradientLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.stop_gradient(inputs)


@keras.utils.register_keras_serializable()
class ConfidenceLossOutputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, assignment_probs, confidence_score):
        if len(confidence_score.shape) == 1:
            confidence_score = tf.expand_dims(confidence_score, axis=-1)

        n_jets = tf.shape(assignment_probs)[1]
        confidence_expanded = tf.tile(
            confidence_score[:, :, tf.newaxis], [1, n_jets, 1]
        )

        output = tf.concat(
            [tf.stop_gradient(assignment_probs), confidence_expanded], axis=-1
        )
        return output

    def get_config(self):
        config = super().get_config()
        return config
