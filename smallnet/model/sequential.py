import sys
from typing import Sequence, Iterator, Tuple

from smallnet.data import DataIterator, Batch
from smallnet.layers.layer import Layer
from smallnet.loss import Loss, MSE
from smallnet.optimiser import Optimiser, SGD
from smallnet.tensor import Tensor


class Sequential:
    def __init__(self, network: Sequence[Layer]) -> None:
        self.loss: Loss = MSE()
        self.optimiser: Optimiser = SGD()
        self.network = network
        self.ready: bool = False

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

    def compile(self, optimizer: Optimiser = SGD(), loss: Loss = MSE()):
        self.optimiser = optimizer
        self.loss = loss
        self.ready = True

    def train(self, inputs: Tensor, targets: Tensor, epochs: int = 1000, iterator: DataIterator = Batch()):
        if not self.ready:
            sys.stderr.write("Model not compiled \n");
        for epochs in range(epochs):
            iter_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.forward(batch.inputs)
                iter_loss += self.loss.loss(batch.targets, predicted)
                grads = self.loss.gradient(batch.targets, predicted)
                self.backward(grads)
                self.optimiser.step(self.parameters_gradients())
                sys.stdout.write(f"loss: {iter_loss} \n")
