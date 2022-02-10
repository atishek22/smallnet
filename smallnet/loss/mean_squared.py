import numpy as np

from smallnet.loss.loss import Loss
from smallnet.tensor import Tensor


class MSE(Loss):
    """
        Mean Squared Error Loss
        Simplifying some Computations
        Would later move to scipy.optimize
    """

    def loss(self, actual: Tensor, predicted: Tensor) -> float:
        diff = predicted - actual
        # return np.sqrt(np.sum(diff ** 2) / len(diff)) # for MSE
        return np.sum(diff ** 2)

    def gradient(self, actual: Tensor, predicted: Tensor) -> Tensor:
        # y = self.loss(actual, predicted)
        diff = predicted - actual
        # return (1/(2 * y)) * 2 * (predicted - actual) # for MSE
        return 2 * diff
