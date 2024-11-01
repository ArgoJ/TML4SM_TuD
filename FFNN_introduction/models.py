"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""


# %%   
"""
Import modules

"""
import tensorflow as tf
import datetime
now = datetime.datetime.now

from keras import layers, Model, Input, Sequential, constraints
from typing import Literal




# %%   
"""
_x_to_y: custom trainable layer

"""

class _x_to_y(layers.Layer):
    def __init__(self):
        super(_x_to_y, self).__init__()
        # define hidden layers with activation functions
        self.ls = [layers.Dense(16, 'softplus')]
        self.ls += [layers.Dense(16, 'softplus')]
        # scalar-valued output function
        self.ls += [layers.Dense(1)]
            
    def call(self, x):     
        
        for l in self.ls:
            x = l(x)
        return x


# %%   
"""
CustomFFNN: custom trainable layer

"""

class CustomFFNN(Model):
    def __init__(
            self, 
            input_size: int,
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            use_derivative: bool = False,
        ) -> None:
        super(CustomFFNN, self).__init__()
        self.use_derivative = use_derivative

        if activations is None:
            activations = ['relu' for _ in range(len(hidden_sizes)-1)]
            activations.append('linear')
        assert len(activations) == len(hidden_sizes), (
            f'Size missmatch between hidden_size ({len(hidden_sizes)}) and activations ({len(activations)})!'
        )

        if non_negs is None:
            non_negs = [False for _ in range(len(hidden_sizes))]
        assert len(non_negs) == len(hidden_sizes), (
            f'Size missmatch between hidden_size ({len(hidden_sizes)}) and non_neg ({len(non_negs)})!'
        )

        # Define the model
        self.model = Sequential()
        self.model.add(Input(shape=(input_size,)))
        for num_neurons, activation, non_neg in zip(hidden_sizes, activations, non_negs):
            self.model.add(layers.Dense(num_neurons, activation, kernel_constraint=constraints.NonNeg if non_neg else None))

            
    def call(self, x) -> tf.Tensor:
        if self.use_derivative:
            return self._compute_gradients(x)
        return self.model(x)
    

    def _compute_gradients(self, inputs) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.model(inputs)
        gradients = tape.gradient(outputs, inputs)
        return tf.concat([outputs, gradients], axis=-1)



# %%   
"""
ICNN: Input convex neural network - creates convex output

"""

class ICNN(CustomFFNN):
    def __init__(
            self, 
            input_size: int,
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
            use_derivative: bool = False,
        ) -> None:
        possible_activations = ['linear', 'softplus', 'relu']
        assert all(activation in possible_activations for activation in activations), (
            f'Activation {activations} cannot be used for ICNN!'
        )

        non_negs = [False] + [True for _ in range(len(hidden_sizes)-1)]
        super(ICNN, self).__init__(input_size, hidden_sizes, activations, non_negs, use_derivative)


# %%   
"""
main: construction of the NN model

"""

def main(**kwargs) -> Model:
    # define input shape
    xs = Input(shape=[1])
    # define which (custom) layers the model uses
    ys = _x_to_y(**kwargs)(xs)
    # connect input and output
    model = Model(inputs = [xs], outputs = [ys])
    # define optimizer and loss function
    model.compile('adam', 'mse')
    return model