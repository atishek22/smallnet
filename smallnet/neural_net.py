from typing import Sequence, Iterator, Tuple

from smallnet.tensor import Tensor
from smallnet.layers.layer import Layer

class NeuralNet:
    def __init__(self, network: Sequence[Layer]) -> None:
        self.network = network

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for the network"""
        for layer in self.network:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grads: Tensor) -> Tensor:
        """Backward pass/ backpropagation for the network"""
        for layer in reversed(self.network):
            grads = layer.backward(grads)
        return grads

    def parameters_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.network:
            for name, parameter in layer.parameters.items():
                yield parameter, layer.gradients[name]


