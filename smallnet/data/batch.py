from typing import Iterator

import numpy as np

from smallnet.data.data import Data, DataIterator
from smallnet.tensor import Tensor


class Batch(DataIterator):
    """"""

    def __init__(self, batch_size: int = 32, shuffled: bool = True):
        self.batch_size = batch_size
        self.shuffled = shuffled

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Data]:
        if len(inputs) != len(targets):
            raise Exception("Invalid inputs and target length")
        indices = np.arange(0, len(inputs), self.batch_size)
        if self.shuffled:
            np.random.shuffle(indices)

        for index in indices:
            last = index + self.batch_size
            yield Data(inputs[index:last], targets[index:last])
