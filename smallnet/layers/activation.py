from abc import ABC, abstractmethod
from typing import Callable

from smallnet.tensor import Tensor
from smallnet.layers.layer import Layer

Function = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
        Activation layer applies a function to each input
    """
    def __init__(self, function: Function, derivative: Function) -> None:
        super().__init__()
        self.function = function
        self.derivative = derivative

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the function to the inputs"""
        self.inputs = inputs
        return self.function(inputs)

    def backward(self, grads: Tensor) -> Tensor:
        """
            backpropagating the errors
        """
        return self.derivative(self.inputs) * grads
