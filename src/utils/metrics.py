import keras as keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class AssignmentAccuracy(keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        score_l0 = y_pred[..., 0]
        score_l1 = y_pred[..., 1]

        # (B, J, J) : broadcasting
        product_matrix = tf.expand_dims(score_l0, axis=2) + tf.expand_dims(
            score_l1, axis=1
        )

        # -----------------------------------
        # 2) Forbid same jet twice
        # -----------------------------------
        J = tf.shape(product_matrix)[1]
        mask = 1.0 - tf.eye(J, batch_shape=[tf.shape(product_matrix)[0]])
        product_matrix = product_matrix * mask

        # -----------------------------------
        # 3) Find best jet pair
        # -----------------------------------
        B = tf.shape(product_matrix)[0]
        J = tf.shape(product_matrix)[1]

        flat = tf.reshape(product_matrix, [B, -1])
        idx = tf.argmax(flat, axis=1, output_type=tf.int32)

        # Convert flat index -> (i, j)
        jet_i = tf.cast(idx // J, tf.int32)
        jet_j = tf.cast(idx % J, tf.int32)

        # indices for lepton 0
        pred_indices = tf.stack([jet_i, jet_j], axis=-1)  # shape: (batch_size, 2)
        true_indices = tf.cast(
            tf.argmax(y_true, axis=1), tf.int32
        )  # shape: (batch_size, 2)

        matches = tf.reduce_all(
            tf.equal(pred_indices, true_indices), axis=-1
        )  # shape: (batch_size,)
        matches = tf.cast(matches, self.dtype)  # shape: (batch_size,)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            matches = matches * sample_weight
            count = tf.reduce_sum(sample_weight)
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(count, self.dtype))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable()
class RegressionDeviation(keras.metrics.Metric):
    """
    Computes mean per-component relative deviation:
        |y_pred - y_true| / max(|y_true|, alpha)
    averaged over batch and variables.
    """

    def __init__(
        self,
        alpha=1.0,
        name="deviation",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)

        # Accumulators
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Compute per-component relative deviation
        denom = tf.maximum(tf.abs(y_true), self.alpha)
        rel = tf.abs(y_pred - y_true) / denom  # shape (batch, n_items, n_vars)
        rel = tf.reduce_mean(rel, axis=[1, 2])  # per-sample scalar

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, rel.dtype)
            rel = rel * sample_weight

        # Accumulate
        self.total.assign_add(tf.reduce_sum(rel))
        self.count.assign_add(tf.cast(tf.size(rel), self.dtype))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class BinnedRegressionAccuracy(keras.metrics.Metric):
    def __init__(self, name="binned_regression_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(
            y_pred, tf.float32
        )  # shape: (batch_size, num_leptons, num_components, n_bins) - probabilities
        # y_true and y_pred are expected to be of shape (batch_size, num_leptons, num_components , n_bins)
        pred_bins = tf.argmax(
            y_pred, axis=-1
        )  # shape: (batch_size, num_leptons, num_components)
        pred_bins = tf.cast(pred_bins, tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        matches = tf.cast(
            tf.equal(y_true, pred_bins), tf.bool
        )  # shape: (batch_size, num_leptons, num_components)
        matches = tf.math.reduce_all(matches, axis=[1, 2])  # shape: (batch_size,)
        matches = tf.cast(matches, self.dtype)  # shape: (batch_size,)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            matches = matches * sample_weight

        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), self.dtype))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()


def _get_metric(metric_name):
    if metric_name not in globals():
        raise ValueError(f"Metric '{metric_name}' not found in source.metrics.")
    return globals()[metric_name]
