# %%   
import tensorflow as tf

from keras import layers, Model, Input, Sequential, constraints
from typing import Literal


# %%   
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

        # Define the model
        self.model = Sequential()
        self.model.add(Input(shape=(input_size,)))
        for num_neurons, activation, non_neg in zip(hidden_sizes, activations, non_negs):
            self.model.add(layers.Dense(
                num_neurons, 
                activation, 
                kernel_constraint=constraints.NonNeg if non_neg else None,
                bias_constraint=constraints.NonNeg if non_neg else None,
            ))

            
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
