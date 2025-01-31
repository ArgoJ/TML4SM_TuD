import tensorflow as tf
from keras import layers, Model, Input
from typing import Literal

from .models import Layer_Sequence


class GSMLayer(layers.Layer):
    def __init__(
            self, 
            hidden_sizes: list[int],
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            g: float = 1.0,
            **kwargs
    ):
        super(GSMLayer, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
        self.g = g
        self.concat = layers.Concatenate(axis=-1)
        self.energy = Layer_Sequence(hidden_sizes, activations, non_negs)

    # @tf.function(reduce_retracing=True)
    def call(self, inputs, states):
        eps_n = inputs[:, :1]
        dt_n = inputs[:, 1:2]
        gamma_old = states[0]
        x = self.concat([eps_n, gamma_old])

        with tf.GradientTape() as tape:
            tape.watch(x)
            e = self.energy(x)
        jac = tape.batch_jacobian(e, x)

        gamma_dot = - self.g * jac[:, 0, :1]
        gamma_new = gamma_old + dt_n * gamma_dot
        return jac[:, 0, 1:2], [gamma_new]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        #   define initial values of the internal variables     
        return [tf.zeros([batch_size, 1], dtype=dtype)]




class GSMModel(Model):
    def __init__(
            self,
            hidden_sizes: list[int],
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
    ) -> None:
        super(GSMModel, self).__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.gsm_layer = layers.RNN(GSMLayer(hidden_sizes, activations, non_negs), return_sequences=True, return_state=False)
    
    def call(self, inputs: tuple | list | tf.Tensor) -> tf.Tensor:
        if isinstance(inputs, (tuple, list)):
            inputs = self.concat(inputs) 
        return self.gsm_layer(inputs)