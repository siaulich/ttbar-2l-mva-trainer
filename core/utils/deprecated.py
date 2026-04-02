import tensorflow as tf
import keras


@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    def __init__(
        self,
        var_weights=None,  # shape (num_vars,) or None
        epsilon=1e-8,  # numerical safety
        name="regression_loss",
        **kwargs,
    ):
        print(
            "WARNING: RegressionLoss is deprecated and will be removed in future versions. Please use RegressionMSE or RegressionMAE instead."
        )
        super().__init__(name=name, **kwargs)
        self.epsilon = float(epsilon)
        self.var_weights = (
            tf.constant(var_weights, dtype=tf.float32)
            if var_weights is not None
            else None
        )

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: shape (batch, n_items, n_vars)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        rel = y_true - y_pred
        sq = tf.square(rel)  # (batch, n_items, n_vars)

        # apply per-variable weights if given
        if self.var_weights is not None:
            # Ensure shape broadcastable: (n_vars,) or (n_vars_of_sq)
            sq = sq * tf.maximum(self.var_weights, self.epsilon)

        # reduce: mean over vars and items, produce per-sample loss
        per_sample = tf.reduce_mean(sq, axis=[1, 2])  # (batch,)

        return per_sample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "var_weights": (
                    None
                    if self.var_weights is None
                    else self.var_weights.numpy().tolist()
                ),
            }
        )
        return config

    def from_config(cls, config):
        print(
            "WARNING: RegressionLoss is deprecated and will be removed in future versions. Please use RegressionMSE or RegressionMAE instead."
        )
        return cls(**config)
