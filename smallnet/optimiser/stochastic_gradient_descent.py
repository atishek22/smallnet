from smallnet.optimiser.optimiser import Optimiser
from smallnet.model import Sequential

# TODO: Generalise for every model later, currently works only for Sequential model
class SGD(Optimiser):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def step(self, network: Sequential) -> None:
        for parameter, gradient in network.parameters_gradients():
            parameter -= self.learning_rate * gradient
