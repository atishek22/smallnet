from abc import ABC, abstractmethod
from smallnet.neural_net import NeuralNet

class Optimiser(ABC):
    @abstractmethod
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, network: NeuralNet) -> None:
        """Take a gradient descent step"""
        raise NotImplementedError
