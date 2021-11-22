import numpy as np

from smallnet.layers.layer import Layer
from smallnet.tensor import Tensor

class Linear(Layer):
    """
        Linear relationship between inputs and outputs
        output = input * weights + biases
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        """Randomly initailise the weights and biases"""
        super().__init__()
        self.parameters['weights'] = np.random.rand(input_size, output_size)
        self.parameters['biases'] = np.random.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
            Output = inputs @ weights + biases
        """
        # storing the inputs for back propogation
        self.inputs = inputs
        return inputs @ self.parameters['weights'] + self.parameters['biases']

    def backward(self, grads: Tensor) -> Tensor:
        """
            Backpropagating the errors (gradient wrt to the inputs for that layer)

            y = f(x)
            x = input @ weights + biases
            dy/d(inputs) = f'(x) @ weights.T
            dy/d(weights) = inputs.T @ f'(x)
            dy/d(biases) = f'(x)
        """
        # from input_size * output_size -> output_size by summing along the rows
        self.gradients['biases'] = np.sum(grads, axis=0)
        self.gradients['weights'] = self.inputs.T @ grads
        return grads @ self.parameters['weights'].T

