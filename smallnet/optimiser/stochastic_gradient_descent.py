from typing import Iterator, Tuple

from smallnet.optimiser.optimiser import Optimiser
from smallnet.tensor import Tensor


# TODO: Generalise for every model later, currently works only for Sequential model
class SGD(Optimiser):
    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def step(self, params_and_grads: Iterator[Tuple[Tensor, Tensor]]) -> None:
        for parameter, gradient in params_and_grads:
            parameter -= self.learning_rate * gradient
