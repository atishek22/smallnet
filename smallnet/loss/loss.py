import numpy as np
from abc import abstractmethod, ABC

from smallnet.tensor import Tensor

class Loss(ABC):

    @abstractmethod
    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        """Calculate the loss value"""
        raise NotImplementedError

    @abstractmethod
    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        """Calculate the gradient for the loss value"""
        raise NotImplementedError

