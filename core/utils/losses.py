import tensorflow as tf
import keras as keras


@keras.utils.register_keras_serializable()
class AssignmentLoss(keras.losses.Loss):
    def __init__(self, lambda_excl=0.0, epsilon=1e-7, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        # y_true: (batch, n_jets, 2)
        # y_pred: (batch, n_jets, 2)
        # IMPORTANT: supply mask in y_true[..., 2] or as sample_weight argument

        # Clip probabilities
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0)

        # ---- Cross entropy ----
        ce = -y_true * tf.math.log(y_pred)
        ce = tf.reduce_mean(ce, axis=[1, 2])  # sum over jets and leptons
        ce_loss = ce

        # ---- Exclusivity penalty ----
        # Penalize p(j,1)*p(j,2) for each jet
        if self.lambda_excl > 0:
            p1 = y_pred[:, :, 0]
            p2 = y_pred[:, :, 1]
            overlap = p1 * p2  # shape (batch, jets)
            overlap = overlap
            excl_loss = self.lambda_excl * tf.reduce_sum(
                overlap, axis=-1
            )  # sum over jets
        else:
            excl_loss = 0.0

        return ce_loss + excl_loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambda_excl": self.lambda_excl, "epsilon": self.epsilon})
        return cfg

@keras.utils.register_keras_serializable()
class RegressionHuber(keras.losses.Loss):
    def __init__(self, delta=1.0, name="regression_huber_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta

    def call(self, y_true, y_pred, sample_weight=None):
        huber = keras.losses.huber(y_true, y_pred, delta=self.delta)
        huber = tf.reduce_mean(huber, axis=[-1])  # mean over items and vars -> (batch,)
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, huber.dtype), [-1])
            huber = huber * sample_weight
        return huber

@keras.utils.register_keras_serializable()
class RegressionMSE(keras.losses.Loss):
    def __init__(
        self,
        name="regression_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: shape (batch, n_items, n_vars)
        """
        sq = tf.square(y_true - y_pred)  # (batch, n_items, n_vars)
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, sq.dtype), [-1, 1, 1])
            sq = sq * sample_weight

        # reduce: mean over vars and items, produce per-sample loss
        per_sample = tf.reduce_mean(sq, axis=[1, 2])  # (batch,)

        return per_sample


class RegressionMAE(keras.losses.Loss):
    def __init__(self, name="regression_mae", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        mae = tf.abs(y_true - y_pred)
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, mae.dtype), [-1, 1, 1])
            mae = mae * sample_weight
        mae = tf.reduce_mean(mae, axis=[1, 2])  # (batch,)
        return mae


class BinnedRegressionLoss(keras.losses.Loss):
    def __init__(self, name="binned_regression_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch, n_items, n_vars, 1) - bin indices
        # y_pred: (batch, n_items, n_vars, n_bins) - probabilities
        n_bins = tf.shape(y_pred)[-1]

        y_true = tf.one_hot(tf.cast(y_true,tf.int32), n_bins, dtype=tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Cross entropy per item and var
        ce = -tf.reduce_sum(
            y_true * tf.math.log(y_pred + 1e-7), axis=-1
        )  # (batch, items, vars)

        ce = tf.reduce_mean(ce, axis=[1, 2])  # mean over items and vars -> (batch,)

        if sample_weight is not None:
            ce = ce * sample_weight

        return ce


@keras.utils.register_keras_serializable()
class PtEtaPhiLoss(keras.losses.Loss):

    def __init__(
        self, w_pt=1.0, w_eta=1.0, w_phi=1.0, w_e=0.0, name="PtEtaPhi_loss", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.w_pt = w_pt
        self.w_eta = w_eta
        self.w_phi = w_phi
        self.w_e = w_e
        self.eps = 1e-8

    def call(self, y_true, y_pred):
        px_t, py_t, pz_t = y_true[..., 0], y_true[..., 1], y_true[..., 2]
        px_p, py_p, pz_p = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]

        # norms
        p_t = tf.sqrt(px_t * px_t + py_t * py_t + pz_t * pz_t + self.eps)
        p_p = tf.sqrt(px_p * px_p + py_p * py_p + pz_p * pz_p + self.eps)

        # transverse momenta
        pt_t = tf.sqrt(px_t * px_t + py_t * py_t + self.eps)
        pt_p = tf.sqrt(px_p * px_p + py_p * py_p + self.eps)

        # -------- Stable pt loss (scale invariant)
        loss_pt = tf.square((pt_t - pt_p) / (pt_t + self.eps))

        # -------- Stable "eta" loss via z-direction
        zdir_t = pz_t / p_t
        zdir_p = pz_p / p_p
        loss_eta = tf.square(zdir_t - zdir_p)

        # -------- Stable phi loss without acos
        cos_dphi = (px_t * px_p + py_t * py_p) / (pt_t * pt_p + self.eps)
        loss_phi = 1.0 - cos_dphi  # smooth, bounded

        # -------- Stable energy / magnitude loss
        loss_e = tf.square((p_t - p_p) / (p_t + p_p + self.eps))

        total = (
            self.w_pt * loss_pt
            + self.w_eta * loss_eta
            + self.w_phi * loss_phi
            + self.w_e * loss_e
        )
        return tf.reduce_mean(total, axis=-1)


@keras.utils.register_keras_serializable()
class MagnitudeDirectionLoss(keras.losses.Loss):
    """
    Loss for two neutrino 3-momenta:
    y_true, y_pred have shape (batch, 2, 3)
    L = w_r * relative magnitude loss + w_theta * angular loss
    """

    def __init__(
        self,
        w_mag=1.0,  # weight for magnitude term
        w_dir=1.0,  # weight for direction term
        epsilon=1e-3,  # numerical safety
        log_mag=False,  # whether to use log-magnitude loss
        name="magdir_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.w_mag = float(w_mag)
        self.w_dir = float(w_dir)
        self.epsilon = float(epsilon)
        self.log_mag = log_mag

    def call(self, y_true, y_pred, sample_weight=None):
        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # ---- Magnitude loss ----
        mag_true = tf.norm(y_true, axis=-1)  # (batch, 2)
        mag_pred = tf.norm(y_pred, axis=-1)  # (batch, 2)

        mag_loss = tf.square(
            (mag_true - mag_pred) / (mag_true + self.epsilon)
        )  # (batch, 2)
        mag_loss = tf.reduce_mean(mag_loss, axis=-1)  # (batch,)

        # ---- Direction loss ----
        dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)  # (batch, 2)
        mag_product = mag_true * mag_pred  # (batch, 2)
        cos_theta = dot_product / (mag_product + self.epsilon)  # (batch, 2)
        theta = tf.acos(cos_theta)  # (batch, 2)
        dir_loss = tf.square(theta)  # (batch, 2)
        dir_loss = tf.reduce_mean(dir_loss, axis=-1)  # (batch,)
        # ---- Total loss ----
        total_loss = self.w_mag * mag_loss + self.w_dir * dir_loss

        # apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, total_loss.dtype), [-1])
            total_loss = total_loss * sample_weight
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "w_mag": self.w_mag,
                "w_dir": self.w_dir,
                "epsilon": self.epsilon,
            }
        )
        return config


import tensorflow as tf
import keras


@keras.utils.register_keras_serializable()
class RestframeLoss(keras.losses.Loss):

    def __init__(self, eps=1e-6, name="restframe_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps

    def to_4vec(self, p3):
        energy = tf.sqrt(tf.reduce_sum(p3**2, axis=-1, keepdims=True) + self.eps)
        return tf.concat([p3, energy], axis=-1)

    def boost(self, p4, beta):
        beta2 = tf.reduce_sum(beta**2, axis=-1, keepdims=True)
        beta2 = tf.clip_by_value(beta2, 0.0, 1.0 - 1e-6)
        gamma = 1.0 / tf.sqrt(1.0 - beta2 + self.eps)

        p_vec = p4[..., :3]
        E = p4[..., 3:4]

        bp = tf.reduce_sum(beta * p_vec, axis=-1, keepdims=True)
        p_parallel = (bp / (beta2 + self.eps)) * beta  # Fixed
        p_perp = p_vec - p_parallel

        boosted_E = gamma * (E + bp)
        boosted_p_parallel = gamma * (p_parallel + beta * E)  # Fixed sign
        boosted_p = boosted_p_parallel + p_perp

        return tf.concat([boosted_p, boosted_E], axis=-1)

    def call(self, y_true, y_pred):
        true_neutrinos = y_true[:, :2, :3]  # (batch,2,3)
        restframe_4vec = y_true[:, 2:, :]  # (batch,2,4)

        pred_neutrinos = y_pred  # (batch,2,3)

        pred_4 = self.to_4vec(pred_neutrinos)
        true_4 = self.to_4vec(true_neutrinos)

        beta = -restframe_4vec[..., :3] / (restframe_4vec[..., 3:4] + self.eps)

        diff = pred_4 - true_4

        diff_boosted = self.boost(diff, beta)

        loss = tf.reduce_mean(tf.square(diff_boosted), axis=[1, 2])
        return loss


class ConfidenceScoreLoss(keras.losses.Loss):
    def __init__(self, epsilon=1e-7, name="confidence_score_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def call(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch, n_jets, 2)
        # y_pred: (batch, n_jets, 3)
        # Get assignment probabilities from y_pred
        assignment_probs = y_pred[:, :, :2]
        confidence_scores = y_pred[:, :, 2]  # shape (batch, n_jets)
        confidence_scores = tf.reduce_max(confidence_scores, axis=-1)  # shape (batch,)

        # Clip probabilities
        assignment_probs = tf.clip_by_value(assignment_probs, self.epsilon, 1.0)

        # Get Assignment per Lepton
        assigned_lepton = tf.argmax(y_true, axis=-2)  # shape (batch, 2)
        pred_lepton = tf.argmax(assignment_probs, axis=-2)  # shape (batch, 2)

        correct_assignment = tf.reduce_all(
            tf.equal(assigned_lepton, pred_lepton), axis=-1
        )
        correct_assignment = tf.cast(correct_assignment, tf.float32)  # shape (batch,)

        # Confidence score loss (Binary Cross Entropy)
        confidence_scores = tf.clip_by_value(confidence_scores, self.epsilon, 1.0)
        bce = -(
            correct_assignment * tf.math.log(confidence_scores)
            + (1.0 - correct_assignment) * tf.math.log(1.0 - confidence_scores)
        )
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, bce.dtype), [-1])
            bce = bce * sample_weight
        return bce


def _get_loss(loss_name):
    if loss_name not in globals():
        raise ValueError(f"Loss '{loss_name}' not found in core.losses.")
    return globals()[loss_name]
