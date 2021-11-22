import numpy as np

from smallnet.tensor import Tensor
from smallnet.layers.activation import Activation

def _sigmoid(x: Tensor) -> Tensor:
    return 1/(1 + np.exp(-x))

def _sigmoid_derivative(x: Tensor) -> Tensor:
    y = _sigmoid(x)
    return y*(1 - y)

class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(_sigmoid, _sigmoid_derivative)
