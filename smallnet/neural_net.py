from typing import Sequence, Iterator, Tuple

from smallnet.tensor import Tensor
from smallnet.layers.layer import Layer
from smallnet.data.data import DataIterator
from smallnet.data.batch import Batch
from smallnet.loss.loss import Loss
from smallnet.loss.mean_squared import MSE
from smallnet.optimiser.optimiser import Optimiser
from smallnet.optimiser.stochastic_gradient_descent import SGD

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
            for (_, parameter), (__, gradient) in zip(layer.parameters.items(), layer.gradients.items()):
                yield parameter, gradient

    def train(self,
              inputs: Tensor,
              targets: Tensor,
              epochs: int = 4000,
              iterator: DataIterator = Batch(),
              loss: Loss = MSE(),
              optimiser: Optimiser = SGD()) -> None:
        for epoch in range(epochs):
            iter_loss = 0.0
            for batch in iterator(inputs, targets):
                predicted = self.forward(batch.inputs)
                iter_loss += loss.loss(batch.targets, predicted)
                grads = loss.gradient(batch.targets, predicted)
                self.backward(grads)
                optimiser.step(self)
            print(epoch, iter_loss)

