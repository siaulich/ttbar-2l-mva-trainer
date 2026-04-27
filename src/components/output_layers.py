import keras as keras


@keras.saving.register_keras_serializable()
class OutputUpScaleLayer(keras.layers.Layer):
    def __init__(self, name="OutputUpScaleLayer", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.std = self.add_weight(
            shape=input_shape[1:],
            initializer="ones",
            trainable=False,
            name="output_std",
        )
        self.mean = self.add_weight(
            shape=input_shape[1:],
            initializer="zeros",
            trainable=False,
            name="output_mean",
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.std + self.mean

    def set_stats(self, mean=None, std=None):
        if mean is not None:
            self.mean.assign(mean)
        if std is not None:
            self.std.assign(std)

    def get_config(self):
        config = super().get_config()
        return config

    def get_stats(self):
        return {"mean": self.mean.numpy(), "std": self.std.numpy()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
