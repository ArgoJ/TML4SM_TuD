import tensorflow as tf
from keras import layers, Model, Input
from typing import Literal

from .models import Layer_Sequence


class GammaEvolutionLayer(layers.Layer):
    def __init__(self, E=2.0, eta=1.0, **kwargs):
        super(GammaEvolutionLayer, self).__init__(**kwargs)
        self.E = tf.constant(E, dtype=tf.float32)
        self.eta = tf.constant(eta, dtype=tf.float32)
        self.state_size = 1  # sigma (stress)
    
    def call(self, inputs, states):
        eps_n = inputs[:, :1]
        dt_n = inputs[:, 1:2]
        gamma_prev = states[0]

        # Explicit Euler update
        gamma_new = gamma_prev + dt_n * (self.E / self.eta) * (eps_n - gamma_prev)
        return gamma_new, [gamma_new]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1])]


class SigmaLayer(layers.Layer):
    def __init__(self, E=2.0, E_infty=0.5, **kwargs):
        super(SigmaLayer, self).__init__(**kwargs)
        self.E = tf.constant(E, dtype=tf.float32)
        self.E_infty = tf.constant(E_infty, dtype=tf.float32)
    
    def call(self, inputs):
        eps_n = inputs[:, :, 0:1]
        gamma_n = inputs[:, :, 1:2]
        sig_n = self.E_infty * eps_n + self.E * (eps_n - gamma_n)
        return sig_n
    

class GammaFFNNLayer(layers.Layer):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
    ):
        super(GammaFFNNLayer, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
        self.concat = layers.Concatenate(axis=-1)
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs, **kwargs)

    def call(self, inputs, states):
        eps_n = inputs[:, :1]
        dt_n = inputs[:, 1:2]
        gamma_old = states[0]
        x = self.concat([eps_n, gamma_old])
        gamma_dot = self.ls(x)
        gamma_new = gamma_old + dt_n * gamma_dot
        return gamma_new, [gamma_new]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        #   define initial values of the internal variables     
        return [tf.zeros([batch_size, 1])]



class MaxwellModel(Model):
    def __init__(self, **kwargs) -> None:
        super(MaxwellModel, self).__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.gamma_layer = layers.RNN(GammaEvolutionLayer(), return_sequences=True, return_state=False)
        self.sigma_layer = SigmaLayer()
    
    def call(self, inputs: tuple | list | tf.Tensor) -> tf.Tensor:
        if isinstance(inputs, (tuple, list)):
            inputs = self.concat(inputs) 
        gamma = self.gamma_layer(inputs)
        sigma_inputs = self.concat([inputs[:, :, :1], gamma])
        return self.sigma_layer(sigma_inputs)


class MaxwellFFNNModel(Model):
    def __init__(
            self,
            hidden_sizes: list[int],
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
    ) -> None:
        super(MaxwellFFNNModel, self).__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.gamma_layer = layers.RNN(GammaFFNNLayer(hidden_sizes, activations, non_negs), return_sequences=True, return_state=False)
        self.sigma_layer = SigmaLayer()
    
    def call(self, inputs: tuple | list | tf.Tensor) -> tf.Tensor:
        if isinstance(inputs, (tuple, list)):
            inputs = self.concat(inputs) 
        gamma = self.gamma_layer(inputs)
        sigma_inputs = self.concat([inputs[:, :, :1], gamma])
        return self.sigma_layer(sigma_inputs)