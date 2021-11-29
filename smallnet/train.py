from smallnet.tensor import Tensor
from smallnet.layers.layer import Layer
from smallnet.data.data import DataIterator
from smallnet.data.batch import Batch
from smallnet.loss.loss import Loss
from smallnet.loss.mean_squared import MSE
from smallnet.optimiser.optimiser import Optimiser
from smallnet.optimiser.stochastic_gradient_descent import SGD
from smallnet.model import Sequential

# later move this back to neural net
def train(network: Sequential,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 4000,
          iterator: DataIterator = Batch(),
          loss: Loss = MSE(),
          optimiser: Optimiser = SGD()) -> None:
    for epoch in range(epochs):
        iter_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = network.forward(batch.inputs)
            iter_loss += loss.loss(batch.targets, predicted)
            grads = loss.gradient(batch.targets, predicted)
            network.backward(grads)
            optimiser.step(network)
        print(epoch, iter_loss)
