# %%   
import tensorflow as tf

from keras import layers, Model, constraints
from typing import Literal
from functools import wraps

from .analytic_potential import (
    get_transversely_isotropic_invariants, 
    get_cubic_anisotropic_invariants,
    get_polyconvex_inputs
)

ICNN_ACTIVATIONS = ['linear', 'softplus', 'relu']

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
                kernel_constraint=constraints.NonNeg() if non_neg else None
            ))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_rank(inputs, 2, message="Inputs should have rank 2 [batch_size, features]")
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

        # Default activations
        if activations is None:
            activations = ['relu'] * (len(hidden_sizes) - 1) + ['linear']
        assert all(activation in ICNN_ACTIVATIONS for activation in activations), (
            f'Activations {activations} cannot be used for ICNN!'
            f'Only {ICNN_ACTIVATIONS} are allowed.'
        )

        non_negs = [False] + [True for _ in range(len(hidden_sizes)-1)]
        super(ICNN_Sequence, self).__init__(hidden_sizes, activations, non_negs, **kwargs)

    
# %%
class Transversely_Isotropic_Invariants_Layer(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = get_transversely_isotropic_invariants(inputs)
        j = invariants[:, 1:2] 
        return tf.concat([invariants, -j], axis=1)
    

# %%
class Cubic_Anisotropic_Invariants_Layer(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = get_cubic_anisotropic_invariants(inputs)
        j = invariants[:, 2:3] 
        return tf.concat([invariants, -j], axis=1)
    

# %%
class Deformation_Layer(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return get_polyconvex_inputs(inputs)


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
class TransIsoInvariantsICNN(InputGradFFNN):
    def __init__(
            self,
            hidden_sizes: list[int], 
            use_derivative: bool = True,
            use_output_and_derivative: bool = False,
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
        ) -> None:

        super(TransIsoInvariantsICNN, self).__init__(use_derivative=use_derivative, use_output_and_derivative=use_output_and_derivative)
        assert all(activation in ICNN_ACTIVATIONS for activation in activations), (
            f'Activations {activations} cannot be used for ICNN!'
            f'Only {ICNN_ACTIVATIONS} are allowed.'
        )
        non_negs = [True for _ in range(len(hidden_sizes))]

        self.invariants_layer = Transversely_Isotropic_Invariants_Layer()
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = self.invariants_layer(inputs)
        out = self.ls(invariants)
        return out
    


# %%   
class CubicAnisoInvariantsICNN(InputGradFFNN):
    def __init__(
            self,
            hidden_sizes: list[int], 
            use_derivative: bool = True,
            use_output_and_derivative: bool = False,
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
        ) -> None:

        super(CubicAnisoInvariantsICNN, self).__init__(use_derivative=use_derivative, use_output_and_derivative=use_output_and_derivative)
        assert all(activation in ICNN_ACTIVATIONS for activation in activations), (
            f'Activations {activations} cannot be used for ICNN!'
            f'Only {ICNN_ACTIVATIONS} are allowed.'
        )
        non_negs = [True for _ in range(len(hidden_sizes))]

        self.invariants_layer = Cubic_Anisotropic_Invariants_Layer()
        self.ls = Layer_Sequence(hidden_sizes, activations, non_negs)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        invariants = self.invariants_layer(inputs)
        out = self.ls(invariants)
        return out
    


# %%   
class DeformationICNN(InputGradFFNN):
    # 
    def __init__(
            self,
            hidden_sizes: list[int], 
            use_derivative: bool = True,
            use_output_and_derivative: bool = False,
            activations: list[Literal['linear', 'softplus', 'relu']] | None = None,
        ) -> None:

        super(DeformationICNN, self).__init__(use_derivative=use_derivative, use_output_and_derivative=use_output_and_derivative)

        self.polyconvex_inputs = Deformation_Layer()
        self.ls = ICNN_Sequence(hidden_sizes, activations)

    def _compute_output(self, inputs: tf.Tensor) -> tf.Tensor:
        polyconvex_inputs = self.polyconvex_inputs(inputs)
        out = self.ls(polyconvex_inputs)
        return out
    




def set_use_output_and_derivative(func):
    @wraps(func)
    def wrapper(model: InputGradFFNN, *args, **kwargs):
        set_back = False

        if not model.use_output_and_derivative:
            model.use_output_and_derivative = True
            model.compile()
            set_back = True

        result = func(model, *args, **kwargs)
        
        if set_back:
            model.use_output_and_derivative = False
            model.compile()
        
        return result
    
    return wrapper