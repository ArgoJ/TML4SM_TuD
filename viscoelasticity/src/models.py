"""
Tutorial Machine Learning in Solid Mechanics (WiSe 23/24)
Task 3: Viscoelasticity
==================
Authors: Dominik K. Klein
         
01/2024
"""


# %%   
import tensorflow as tf

from keras import layers, Model, constraints
from typing import Literal
from functools import wraps
    

ICNN_ACTIVATIONS = ['linear', 'softplus', 'relu']

class Layer_Sequence(layers.Layer):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
        ) -> None:
        super(Layer_Sequence, self).__init__(**kwargs)

        # Default activations and check length
        if activations is None:
            activations = ['relu' for _ in range(len(hidden_sizes)-1)]
            activations.append('linear')
        assert len(activations) == len(hidden_sizes), (
            f'Size missmatch between hidden_size ({len(hidden_sizes)}) and activations ({len(activations)})!'
        )

        # Default non negatives and check length
        if non_negs is None:
            non_negs = [False for _ in range(len(hidden_sizes))]
        assert len(non_negs) == len(hidden_sizes), (
            f'Size missmatch between hidden_size ({len(hidden_sizes)}) and non_neg ({len(non_negs)})!'
        )
        self.ls = []
        for num_neurons, activation, non_neg in zip(hidden_sizes, activations, non_negs):
            self.ls.append(layers.Dense(
                num_neurons, 
                activation, 
                kernel_constraint=constraints.NonNeg() if non_neg else None
            ))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_rank(inputs, 2, message="Inputs should have rank 2 [batch_size, features]")
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x


class RNNCell(layers.Layer):
    
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
    ):
        super(RNNCell, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1
     
        self.concat = layers.Concatenate(axis=1)
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs, **kwargs)

    def call(self, inputs, states):
        #   states are the internal variables
        #   n: current time step, N: old time step  
        eps_n = inputs[:, :1]
        hs = inputs[:, 1:2]
        
        #   gamma: history variable
        gamma_N = states[0]
        
        #   x contains the current strain, the current time step size, and the 
        #   history variable from the previous time step
        x = self.concat([eps_n, hs, gamma_N])
    
        #   x gets passed to a FFNN which yields the current stress and history
        #   variable
        x = self.ls(x)
        sig_n = x[:,0:1]
        gamma_n = x[:,1:2]
        return sig_n , [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        #   define initial values of the internal variables     
        return [tf.zeros([batch_size, 1])]


class RNNModel(Model):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            **kwargs
    ):
        super(RNNModel, self).__init__(**kwargs)
        self.concat = layers.Concatenate(axis=-1)
        self.cell = RNNCell(hidden_sizes, activations, non_negs)
        self.rnn_layer = layers.RNN(self.cell, return_sequences=True, return_state=False)
    
    def call(self, inputs: tuple | list | tf.Tensor) -> tf.Tensor:
        if isinstance(inputs, (tuple, list)):
            inputs = self.concat(inputs) 
        return self.rnn_layer(inputs)