import numpy as np

from smallnet.tensor import Tensor
from smallnet.layers.activation import Activation

def _relu(x: Tensor) -> Tensor:
    return np.maximum(0,x)

def _relu_derivative(x: Tensor) -> Tensor:
    return (x > 0) * 1.0

class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(_relu, _relu_derivative)
