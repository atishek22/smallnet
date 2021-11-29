from abc import ABC, abstractmethod
from smallnet.model import Sequential

class Optimiser(ABC):
    @abstractmethod
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, network: Sequential) -> None:
        """Take a gradient descent step"""
        raise NotImplementedError
