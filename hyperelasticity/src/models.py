# %%   
import tensorflow as tf

from keras import layers, Model, Input, Sequential, constraints
from typing import Literal
from abc import ABC

from .analytic_potential import get_invariants


# %%
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
                kernel_constraint=constraints.NonNeg if non_neg else None
            ))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.ls:
            x = layer(x)
        return x


# %%
class ICNN_Sequence(Layer_Sequence):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
            **kwargs
        ) -> None:
        possible_activations = ['linear', 'softplus', 'relu']

        # Default activations
        if activations is None:
            activations = ['relu' for _ in range(len(hidden_sizes) - 1)]
            activations.append('linear')
        assert all(activation in possible_activations for activation in activations), (
            f'Activations {activations} cannot be used for ICNN!'
            f'Only {possible_activations} are allowed.'
        )

        non_negs = [False] + [True for _ in range(len(hidden_sizes)-1)]
        super(ICNN_Sequence, self).__init__(hidden_sizes, activations, non_negs, **kwargs)

    
# %%
class Invariants_Layer(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = get_invariants(inputs)
        j = invariants[:, 1:2] 
        return tf.concat([invariants, -j], axis=1)



# %%   
class InputGradFFNN(Model):
    def __init__(
            self, 
            use_derivative: bool = False,
            use_output_and_derivative: bool = False
        ) -> None:
        super(InputGradFFNN, self).__init__()
        self.use_output_and_derivative = use_output_and_derivative
        self.use_derivative = False if self.use_output_and_derivative else use_derivative
            
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.use_output_and_derivative:
            return self._compute_output_and_gradient(inputs)
        if self.use_derivative and not self.use_output_and_derivative:
            return self._compute_gradients(inputs)
        return self._compute_output(inputs)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError('_compute_output function need to be implemented!')
    
    def _compute_gradients(self, inputs) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self._compute_output(inputs)
        gradients = tape.gradient(outputs, inputs)
        return gradients
    
    def _compute_output_and_gradient(self, inputs) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self._compute_output(inputs)
        gradients = tape.gradient(outputs, inputs)
        return outputs, gradients
    

# %%
class CustomFFNN(InputGradFFNN):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'tanh', 'relu', 'sigmoid']] | None = None,
            non_negs: list[bool] | None = None,
            use_derivative: bool = False
        ) -> None:
        super(CustomFFNN, self).__init__(use_derivative=use_derivative)
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.ls(inputs)


# %%   
class ICNN(InputGradFFNN):
    def __init__(
            self, 
            hidden_sizes: list[int], 
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
            use_derivative: bool = False,
        ) -> None:
        super(ICNN, self).__init__(use_derivative)
        self.ls = ICNN_Sequence(hidden_sizes, activations)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.ls(inputs)


# %%   
class InvariantsICNN(InputGradFFNN):
    def __init__(
            self,
            hidden_sizes: list[int], 
            use_derivative: bool = True,
            use_output_and_derivative: bool = False,
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
        ) -> None:

        super(InvariantsICNN, self).__init__(use_derivative=use_derivative, use_output_and_derivative=use_output_and_derivative)
        non_negs = [True for _ in range(len(hidden_sizes))]

        self.invariants_layer = Invariants_Layer()
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = self.invariants_layer(inputs)
        out = self.ls(invariants)
        return out