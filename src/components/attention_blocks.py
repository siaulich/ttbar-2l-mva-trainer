"""
This module contains various attention block implementations for use in neural networks.
The attention blocks include:
- SelfAttentionBlock: A self-attention mechanism that allows the model to focus on different parts
  of the input sequence.
- SelfAttentionStack: A stack of self-attention blocks for deeper attention mechanisms.
- MultiHeadAttentionBlock: A multi-head attention mechanism that allows the model to jointly attend
  to information from different representation subspaces.
- MultiHeadAttentionStack: A stack of multi-head attention blocks for more complex attention patterns.
- CrossAttentionBlock: A cross-attention mechanism that attends two sequences to each other.
- CrossAttentionStack: A stack of cross-attention blocks for complex interactions between two sequences.
- PoolingAttentionBlock: An attention mechanism that pools information from a set of seed vectors.
- InducedSetAttentionBlock: An attention mechanism that uses induced sets for efficient attention computation.
- point_transformer: A transformer-like architecture for processing point cloud data.
"""

import keras as keras
import tensorflow as tf
from keras import layers, regularizers


@keras.utils.register_keras_serializable()
class MultiHeadAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        ff_dim=None,
        dropout_rate=0,
        self_attention=False,
        pre_ln=False,
        regularizer=None,
        **kwargs,
    ):
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        super().__init__(**kwargs)
        self.self_attention = self_attention
        self.pre_ln = pre_ln
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim or key_dim * 2
        self.dropout_rate = dropout_rate
        self.supports_masking = False
        # LayerNorms
        self.ln_q = layers.LayerNormalization(epsilon=1e-6, name="ln_q")
        self.ln_kv = (
            layers.LayerNormalization(epsilon=1e-6, name="ln_kv")
            if not self_attention and pre_ln
            else None
        )
        self.ln_ffn = (
            layers.LayerNormalization(epsilon=1e-6, name="ln_ffn") if pre_ln else None
        )

        # Attention
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim // num_heads,
            dropout=dropout_rate,
            name="multi_head_attention",
        )
        self.dropout_attn = layers.Dropout(dropout_rate, name="dropout_attn")

        # Feedforward network
        self.ffn_dense_1 = layers.Dense(
            self.ff_dim, activation="relu", name="ffn_dense_1"
        )
        self.ffn_dense_2 = layers.Dense(key_dim, name="ffn_dense_2")
        self.ffn_dropout = layers.Dropout(dropout_rate, name="ffn_dropout_1")

    def call(
        self,
        query,
        value=None,
        key=None,
        attention_mask=None,
        query_mask=None,
        key_mask=None,
        value_mask=None,
        training=None,
    ):
        if value is None and not self.self_attention:
            raise ValueError("Value must be provided unless self-attention is used.")
        if self.pre_ln:
            return self._call_pre_ln(
                query=query,
                value=value,
                key=key,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                attention_mask=attention_mask,
                training=training,
            )
        else:
            return self._call_post_ln(
                query=query,
                value=value,
                key=key,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                attention_mask=attention_mask,
                training=training,
            )

    def _call_pre_ln(
        self,
        query,
        value,
        key,
        query_mask,
        value_mask,
        key_mask,
        attention_mask,
        training,
    ):
        if key is None:
            key = value
            key_mask = value_mask

        if self.self_attention:
            q_norm = self.ln_q(query)
            k_norm = v_norm = q_norm
            # For self-attention, use query masks for key/value too
            key_mask = value_mask = query_mask
        else:
            q_norm = self.ln_q(query)
            if self.ln_kv is not None:
                k_norm = self.ln_kv(key)
                v_norm = self.ln_kv(value)
            else:
                k_norm = key
                v_norm = value

        attn_out = self.attn(
            q_norm,
            k_norm,
            v_norm,
            query_mask=query_mask,
            key_mask=key_mask,
            value_mask=value_mask,
            attention_mask=attention_mask,
            training=training,
        )
        attn_out = self.dropout_attn(attn_out, training=training)
        x = query + attn_out  # residual

        if self.ln_ffn is not None:
            ffn_in = self.ln_ffn(x)
        else:
            ffn_in = x

        ffn_out = self.ffn_dense_1(ffn_in)
        ffn_out = self.ffn_dense_2(ffn_out)
        ffn_out = self.ffn_dropout(ffn_out, training=training)

        return x + ffn_out

    def _call_post_ln(
        self,
        query,
        value,
        key,
        query_mask,
        value_mask,
        key_mask,
        attention_mask,
        training,
    ):
        if key is None:
            key = value
            key_mask = value_mask
        # Attention
        if self.self_attention:
            attn_out = self.attn(
                query,
                query,
                query,
                query_mask=query_mask,
                key_mask=query_mask,
                value_mask=query_mask,
                attention_mask=attention_mask,
                training=training,
            )
        else:
            attn_out = self.attn(
                query,
                key,
                value,
                query_mask=query_mask,
                key_mask=key_mask,
                value_mask=value_mask,
                training=training,
            )

        attn_out = self.dropout_attn(attn_out, training=training)
        x = self.ln_q(query + attn_out)  # LN after residual

        # FFN
        ffn_out = self.ffn_dense_1(x)
        ffn_out = self.ffn_dense_2(ffn_out)
        ffn_out = self.ffn_dropout(ffn_out, training=training)

        if self.ln_ffn is not None:
            return self.ln_ffn(x + ffn_out)
        else:
            return x + ffn_out

    def build(self, query_shape, value_shape, key_shape=None):
        """Fixed build method for Keras compatibility"""
        if key_shape is None:
            key_shape = value_shape
        # Build layers
        self.ln_q.build(query_shape)
        if self.ln_kv is not None:
            self.ln_kv.build(value_shape)
        if self.ln_ffn is not None:
            self.ln_ffn.build(query_shape)

        self.attn.build(query_shape, value_shape, key_shape)
        self.ffn_dense_1.build(
            (None, None, self.key_dim)
        )  # Use None for flexible batch/seq dims
        self.ffn_dense_2.build((None, None, self.ff_dim))

        super().build(query_shape)

    def compute_output_shape(self, query_shape, key_shape=None, value_shape=None):
        if value_shape is None:
            value_shape = query_shape
        if key_shape is None:
            key_shape = query_shape
        return (query_shape[0], query_shape[1], self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "self_attention": self.self_attention,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)


@keras.utils.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout_rate=0.0,
        ff_dim=None,
        name="self_attention_block",
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim or key_dim * 2
        self.pre_ln = pre_ln
        self.block_name = name

        self.attention = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            name=f"{name}_attention",
            regularizer=regularizer,
            self_attention=True,
            pre_ln=pre_ln,
        )

    def call(self, inputs, mask=None, training=None):
        x = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            key_mask=mask,
            value_mask=mask,
            query_mask=mask,
            training=training,
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "name": self.block_name,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)

    def build(self, input_shape):
        # build all child layers explicitly
        self.attention.build(input_shape, input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def count_params(self):
        return self.attention.count_params()


@keras.utils.register_keras_serializable()
class SelfAttentionStack(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        stack_size=3,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(SelfAttentionStack, self).__init__(**kwargs)
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.stack_size = stack_size
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.pre_ln = pre_ln
        self.attention_blocks = [
            SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"attention_block_{i+1}",
                pre_ln=pre_ln,
                regularizer=regularizer,
            )
            for i in range(stack_size)
        ]

    def call(self, inputs, mask=None, training=None):
        x = inputs
        for block in self.attention_blocks:
            x = block(x, mask=mask, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "stack_size": self.stack_size,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)

    def build(self, input_shape):
        x_shape = input_shape
        for block in self.attention_blocks:
            block.build(x_shape)
        super().build(input_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


@keras.utils.register_keras_serializable()
class MultiHeadAttentionStack(layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        stack_size=3,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        self_attention=False,
        **kwargs,
    ):
        super(MultiHeadAttentionStack, self).__init__(**kwargs)
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.stack_size = stack_size
        self.self_attention = self_attention
        self.pre_ln = pre_ln

        self.attention_blocks = [
            MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout_rate=dropout_rate,
                ff_dim=ff_dim,
                name=f"multi_head_attention_block_{i+1}",
                regularizer=regularizer,
                pre_ln=pre_ln,
                self_attention=self_attention,
            )
            for i in range(stack_size)
        ]

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        key_mask=None,
        value_mask=None,
        training=None,
    ):
        if key is None:
            key = value
        x = query
        if self.self_attention:
            value = query
            key = query
            key_mask = query_mask
            value_mask = query_mask
        for block in self.attention_blocks:
            x = block(
                query=x,
                value=value,
                key=key,
                query_mask=query_mask,
                key_mask=key_mask,
                value_mask=value_mask,
                training=training,
            )
        return x

    def compute_output_shape(self, query_shape, key_shape=None, value_shape=None):
        if value_shape is None:
            value_shape = query_shape
        if key_shape is None:
            key_shape = query_shape
        return (query_shape[0], query_shape[1], self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "stack_size": self.stack_size,
                "self_attention": self.self_attention,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)

    def build(self, query_shape, value_shape):
        super(MultiHeadAttentionStack, self).build(query_shape)
        for block in self.attention_blocks:
            block.build(query_shape, value_shape)

    def count_params(self):
        return sum(block.count_params() for block in self.attention_blocks)


@keras.utils.register_keras_serializable()
class CrossAttentionAssignment(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, query_shape, value_shape, cross_values_shape):
        super().build(query_shape)
        if (
            len(query_shape) != 3
            or len(value_shape) != 3
            or len(cross_values_shape) != 3
        ):
            raise ValueError(
                "Expected all inputs to be rank 3 (batch_size, seq_len, features)"
            )
        self.feature_dim = query_shape[-1]
        self.query_proj = layers.Dense(
            self.feature_dim // 2, use_bias=False, name="query_proj"
        )
        self.value_proj = layers.Dense(
            self.feature_dim // 2, use_bias=False, name="value_proj"
        )
        self.output_proj = layers.Dense(1, use_bias=False, name="output_proj")

    def call(
        self,
        query,
        value,
        cross_values,
        query_mask=None,
        value_mask=None,
        training=None,
    ):

        q_proj = self.query_proj(query)  # (B, N_q, feature_dim//3)
        v_proj = self.value_proj(value)  # (B, N_v, feature_dim//3)
        cv_proj = cross_values  # (B, N_cv, feature_dim//3)

        q_proj = tf.expand_dims(q_proj, axis=2)  # (B, N_q, 1, feature_dim//3)
        v_proj = tf.expand_dims(v_proj, axis=1)  # (B, 1, N_v, feature_dim//3)

        # Compute attention scores
        pair_vectors = tf.concat(
            [q_proj, v_proj, cv_proj], axis=-1
        )  # (B, N_q, N_v, feature_dim)
        scores = self.output_proj(pair_vectors)  # (B, N_q, N_v, 1)
        scores = tf.squeeze(scores, axis=-1)  # (B, N_q, N_v)
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)
            scores = tf.where(
                query_mask[:, :, tf.newaxis],
                scores,
                tf.constant(-1e9, dtype=scores.dtype),
            )
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)
            scores = tf.where(
                value_mask[:, tf.newaxis, :],
                scores,
                tf.constant(-1e9, dtype=scores.dtype),
            )
        return tf.nn.softmax(scores, axis=1)  # (B, N_q, N_v)


@keras.utils.register_keras_serializable()
class JetLeptonAssignment(layers.Layer):
    def __init__(self, dim, use_bias=True, temperature=None, **kwargs):
        """
        Args:
            dim: latent projection dimension
            use_bias: whether to use bias in dense projections
            temperature: optional scaling (default = sqrt(dim))
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.use_bias = use_bias
        self._temperature_param = temperature
        self.temperature = temperature or tf.math.sqrt(tf.cast(dim, tf.float32))

    def build(self, jets_shape, leptons_shape):
        self.query_proj = layers.Dense(self.dim, use_bias=self.use_bias)
        self.key_proj = layers.Dense(self.dim, use_bias=self.use_bias)
        super().build([jets_shape, leptons_shape])

    def call(self, jets, leptons, jet_mask=None):
        """
        Args:
            jets: (B, N_j, d_j)
            leptons: (B, N_l, d_l)
            jet_mask: (B, N_j), boolean or {0,1}
        Returns:
            probs: (B, N_l, N_j) assignment probabilities (softmax over jets)
        """
        # Project into latent space
        queries = self.query_proj(leptons)  # (B, N_l, dim)
        keys = self.key_proj(jets)  # (B, N_j, dim)

        # Compute scaled dot product logits
        logits = tf.einsum("bld,bjd->bjl", queries, keys)  # (B, N_j, N_l)
        logits = logits / self.temperature

        # Apply jet mask (broadcast to match (B, N_l, N_j))
        if jet_mask is not None:
            mask = tf.cast(jet_mask[:, :, tf.newaxis], tf.bool)  # (B, 1, N_j)
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask, logits, neg_inf)

        # Softmax over jets (for each lepton)
        probs = tf.nn.softmax(logits, axis=1)  # (B, N_j, N_l)
        return probs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "use_bias": self.use_bias,
                "temperature": self._temperature_param,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class PoolingAttentionBlock(layers.Layer):
    """
    Set Transformer style Pooling by Multihead Attention (PMA)
    Based on equation (11): PMA_k(Z) = MAB(S, rFF(Z))
    """

    def __init__(
        self,
        key_dim,
        num_seeds,
        num_heads=4,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(PoolingAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.pre_ln = pre_ln

        # Learnable seed vectors (S in the paper)
        self.seed_vectors = self.add_weight(
            shape=(num_seeds, key_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="seed_vectors",
            regularizer=self.regularizer,
        )

        if ff_dim is None:
            self.ff_dim = key_dim * 2
        else:
            self.ff_dim = ff_dim

        # Pre-processing feed-forward for inputs (rFF(Z) in equation 11)
        self.input_ff_1 = layers.Dense(
            self.ff_dim,
            activation="relu",
            name="input_ff_1",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )
        self.input_ff_2 = layers.Dense(
            key_dim,
            name="input_ff_2",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
        )

        # Use MultiHeadAttentionBlock which already implements MAB
        self.MHA = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=self.regularizer,
            pre_ln=pre_ln,
            name="pooling_mha",
        )

    def build(self, input_shape):
        super(PoolingAttentionBlock, self).build(input_shape)

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch_size, seq_len, features). Got {input_shape}."
            )

        if input_shape[-1] != self.key_dim:
            raise ValueError(
                f"Input feature dimension {input_shape[-1]} must match key_dim {self.key_dim}"
            )

        # Build the components
        seed_shape = (None, self.num_seeds, self.key_dim)
        processed_input_shape = (None, input_shape[1], self.key_dim)

        self.input_ff_1.build(input_shape)
        self.input_ff_2.build((*input_shape[:-1], self.ff_dim))
        self.MHA.build(seed_shape, processed_input_shape)

    def call(self, inputs, mask=None, attention_mask=None, training=None):
        """
        Implements PMA_k(Z) = MAB(S, rFF(Z)) from Set Transformer paper

        Args:
            inputs: (batch_size, seq_len, key_dim)
            mask: (batch_size, seq_len) boolean mask for inputs

        Returns:
            output: (batch_size, num_seeds, key_dim)
        """
        batch_size = tf.shape(inputs)[0]

        processed_inputs = self.input_ff_1(inputs)
        processed_inputs = self.input_ff_2(processed_inputs)

        seed_vectors_expanded = tf.broadcast_to(
            self.seed_vectors[tf.newaxis, ...],
            [batch_size, self.num_seeds, self.key_dim],
        )

        # MAB(S, rFF(Z)) using MultiHeadAttentionBlock
        output = self.MHA(
            query=seed_vectors_expanded,
            value=processed_inputs,
            key=processed_inputs,
            key_mask=mask,
            attention_mask=attention_mask,
            training=training,
        )

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_seeds, self.key_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_seeds": self.num_seeds,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)

    def count_params(self):
        return (
            self.seed_vectors.shape[0] * self.seed_vectors.shape[1]
            + self.input_ff_1.count_params()
            + self.input_ff_2.count_params()
            + self.MHA.count_params()
        )


@keras.utils.register_keras_serializable()
class InducedSetAttentionBlock(layers.Layer):
    def __init__(
        self,
        key_dim,
        num_seeds,
        num_heads=4,
        dropout_rate=0.0,
        ff_dim=None,
        regularizer=None,
        pre_ln=True,
        **kwargs,
    ):
        super(InducedSetAttentionBlock, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.ff_dim = ff_dim
        self.regularizer = regularizers.get(regularizer) if regularizer else None
        self.pre_ln = pre_ln

        self.seed_vectors = self.add_weight(
            shape=(num_seeds, key_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="seed_vectors",
            regularizer=self.regularizer,
        )
        self.IMAB = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=self.regularizer,
            pre_ln=pre_ln,
            name="induced_set_attention_mha",
        )
        self.MAB = MultiHeadAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            regularizer=self.regularizer,
            pre_ln=pre_ln,
            name="induced_set_attention_mab",
        )

    def build(self, input_shape):
        super(InducedSetAttentionBlock, self).build(input_shape)
        seed_vectors_shape = (None, self.num_seeds, self.key_dim)

        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input shape (batch_size, num_points, key_dim). Got {input_shape}."
            )

        value_shape = (None, input_shape[1], self.key_dim)

        self.IMAB.build(seed_vectors_shape, value_shape)
        self.MAB.build(value_shape, seed_vectors_shape)

        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.input_spec = layers.InputSpec(shape=input_shape)

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]

        seed_vectors_expanded = tf.broadcast_to(
            self.seed_vectors[tf.newaxis, ...],
            [batch_size, self.num_seeds, self.key_dim],
        )

        induced_output = self.IMAB(
            query=seed_vectors_expanded,
            value=inputs,
            key=inputs,
            key_mask=mask,
            training=training,
        )

        output = self.MAB(
            query=inputs,
            value=induced_output,
            key=induced_output,
            query_mask=mask,
            training=training,
        )

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "num_seeds": self.num_seeds,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "ff_dim": self.ff_dim,
                "pre_ln": self.pre_ln,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        regularizer_config = config.pop("regularizer", None)
        if regularizer_config:
            config["regularizer"] = regularizers.deserialize(regularizer_config)
        return cls(**config)

    def count_params(self):
        return (
            self.seed_vectors.shape[0] * self.seed_vectors.shape[1]
            + self.IMAB.count_params()
            + self.MAB.count_params()
        )
