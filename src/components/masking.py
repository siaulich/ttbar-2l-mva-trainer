import keras as keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class GenerateMask(keras.layers.Layer):
    def __init__(self, padding_value=-999, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, inputs):
        # For shape (batch, seq_len, dim), we reduce across dim to get (batch, seq_len)
        not_pad = tf.not_equal(inputs, self.padding_value)
        mask = tf.reduce_any(not_pad, axis=-1)
        return tf.cast(mask, tf.bool)

    def get_config(self):
        config = super().get_config()
        config.update({"padding_value": self.padding_value})
        return config


@keras.utils.register_keras_serializable()
class TemporalSoftmax(keras.layers.Layer):
    """
    A custom Keras layer that applies a softmax operation along a specified axis,
    with optional masking support. This layer is useful for temporal data where
    masking is required to ignore certain time steps during the softmax computation.
    Attributes:
        axis (int): The axis along which the softmax operation is applied. Defaults to -2.
        supports_masking (bool): Indicates that this layer supports masking.
    Methods:
        call(inputs, mask=None):
            Applies the softmax operation to the inputs along the specified axis.
            If a mask is provided, it is used to ignore certain elements during
            the computation.
        compute_output_shape(input_shape):
            Computes and returns the output shape of the layer, which is the same
            as the input shape.
        get_config():
            Returns the configuration of the layer as a dictionary, including the
            axis attribute.
    Args:
        axis (int, optional): The axis along which to apply the softmax. Defaults to -2.
        **kwargs: Additional keyword arguments for the base Keras Layer class.
    Example:
        ```python
        # Example usage of TemporalSoftmax
        temporal_softmax = TemporalSoftmax(axis=-1)
        inputs = tf.random.uniform((2, 3, 4))
        mask = tf.constant([[1, 1, 0], [1, 0, 0]], dtype=tf.float32)
        outputs = temporal_softmax(inputs, mask=mask)
        print(outputs)
        ```
    """

    def __init__(self, axis=-2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.broadcast_to(mask, tf.shape(inputs))
            not_mask = 1.0 - mask
            inputs += not_mask * -1e9
        softmax_output = tf.nn.softmax(inputs, axis=self.axis)
        if mask is not None:
            softmax_output *= mask
        return softmax_output

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config
