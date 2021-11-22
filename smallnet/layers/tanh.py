import numpy as np

from smallnet.tensor import Tensor
from smallnet.layers.activation import Activation, Function

def _tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def _tanh_derivative(x: Tensor) -> Tensor:
    return 1 - _tanh(x) ** 2

class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(_tanh, _tanh_derivative)


