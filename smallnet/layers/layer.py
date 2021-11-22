from abc import ABC, abstractmethod
from typing import Dict

from smallnet.tensor import Tensor

class Layer(ABC):
    """Abstract definition for a Layer."""

    @abstractmethod
    def __init__(self) -> None:
        self.parameters: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for a layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grads: Tensor) -> Tensor:
        """Backward pass with the errors for the layer"""
        raise NotImplementedError
