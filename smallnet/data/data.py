import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator
from dataclasses import dataclass

from smallnet.tensor import Tensor

@dataclass
class Data:
    inputs: Tensor
    targets: Tensor

class DataIterator(ABC):

    @abstractmethod
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Data]:
        raise NotImplementedError
