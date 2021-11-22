from smallnet.optimiser.optimiser import Optimiser
from smallnet.neural_net import NeuralNet

class SGD(Optimiser):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def step(self, network: NeuralNet) -> None:
        for parameter, gradient in network.parameters_gradients():
            parameter -= self.learning_rate * gradient
