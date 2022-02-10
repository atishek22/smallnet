from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from smallnet.tensor import Tensor


class Optimiser(ABC):
    @abstractmethod
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, params_and_grads: Iterator[Tuple[Tensor, Tensor]]) -> None:
        """Take a gradient descent step"""
        raise NotImplementedError
