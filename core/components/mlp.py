import keras as keras
from keras import regularizers


@keras.utils.register_keras_serializable()
class MLP(keras.layers.Layer):
    """
    Flexible Multi-Layer Perceptron (MLP) layer with configurable architecture.

    Args:
        output_dim: Output dimensionality
        hidden_dims: List of hidden layer dimensions, or 'auto' for geometric interpolation
        num_layers: Number of layers (only used if hidden_dims='auto')
        layer_norm: Whether to apply layer normalization after each layer
        activation: Activation function for output layer
        hidden_activation: Activation function for hidden layers
        dropout_rate: Dropout rate (0 to disable)
        use_bias: Whether to use bias in dense layers
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias weights
        activity_regularizer: Regularizer for layer activations
        apply_final_norm: Whether to apply layer norm after output layer
        apply_final_dropout: Whether to apply dropout after output layer
    """

    def __init__(
        self,
        output_dim,
        hidden_dims="auto",
        num_layers=3,
        layer_norm=False,
        activation="linear",
        hidden_activation="relu",
        dropout_rate=0.0,
        use_bias=True,
        name="mlp",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.activation = activation
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.layers_list = []
        self._built_hidden_dims = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Determine hidden layer dimensions
        if self.hidden_dims == "auto":
            if self.num_layers > 1:
                # Geometric interpolation between input and output dimensions
                hidden_dims_list = [
                    int(
                        input_dim
                        * ((self.output_dim / input_dim) ** (i / self.num_layers))
                    )
                    for i in range(1, self.num_layers)
                ]
            else:
                hidden_dims_list = []
        elif isinstance(self.hidden_dims, (list, tuple)):
            hidden_dims_list = list(self.hidden_dims)
        else:
            raise ValueError(
                f"hidden_dims must be 'auto' or a list/tuple, got {self.hidden_dims}"
            )

        # Store the computed hidden dims for serialization
        self._built_hidden_dims = hidden_dims_list
        self._build_input_shape = input_shape

        # Build hidden layers
        for i, width in enumerate(hidden_dims_list):
            dense_layer = keras.layers.Dense(
                width,
                activation=self.hidden_activation,
                use_bias=self.use_bias,
                name=f"dense_{i}",
            )
            self.layers_list.append(dense_layer)

            if self.layer_norm:
                ln_layer = keras.layers.LayerNormalization(name=f"ln_{i}")
                self.layers_list.append(ln_layer)

            if self.dropout_rate > 0:
                dropout_layer = keras.layers.Dropout(
                    self.dropout_rate, name=f"dropout_{i}"
                )
                self.layers_list.append(dropout_layer)

        # Output layer
        output_layer = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            name="output",
        )
        self.layers_list.append(output_layer)

        # Build all sublayers with the input shape
        x_shape = input_shape
        for layer in self.layers_list:
            if not layer.built:
                layer.build(x_shape)
            x_shape = layer.compute_output_shape(x_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "hidden_dims": self.hidden_dims,
                "num_layers": self.num_layers,
                "layer_norm": self.layer_norm,
                "activation": self.activation,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
            }
        )
        return config


    @classmethod
    def from_config(cls, config):
        # Deserialize regularizers
        kernel_reg = config.pop("kernel_regularizer", None)
        bias_reg = config.pop("bias_regularizer", None)
        activity_reg = config.pop("activity_regularizer", None)
        config.pop("apply_final_norm", None)
        config.pop("apply_final_dropout", None)


        return cls(**config)


@keras.utils.register_keras_serializable()
class EmbeddingMLP(keras.layers.Layer):
    """
    MLP layer that includes an embedding layer for categorical features.

    Args:
        output_dim: Output dimensionality
        hidden_dims: List of hidden layer dimensions, or 'auto' for geometric interpolation
        num_layers: Number of layers (only used if hidden_dims='auto')
        layer_norm: Whether to apply layer normalization after each layer
        activation: Activation function for output layer
        hidden_activation: Activation function for hidden layers
        dropout_rate: Dropout rate (0 to disable)
        use_bias: Whether to use bias in dense layers
        kernel_regularizer: Regularizer for kernel weights
        bias_regularizer: Regularizer for bias weights
        activity_regularizer: Regularizer for layer activations
        apply_final_norm: Whether to apply layer norm after output layer
        apply_final_dropout: Whether to apply dropout after output layer
    """

    def __init__(
        self,
        output_dim,
        hidden_dims="auto",
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.ff_dense_1 = keras.layers.Dense(
            2*self.output_dim,
            activation="relu",
            name="ff_dense_1",
        )
        self.ff_dense_2 = keras.layers.Dense(
            self.output_dim,
            activation="linear",
            name="ff_dense_2",
        )
        self.ff_dropout = keras.layers.Dropout(self.dropout_rate, name="ff_dropout")
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.ff_dense_1(x)
        x = self.ff_dense_2(x)
        x = self.ff_dropout(x, training=training)
        return x