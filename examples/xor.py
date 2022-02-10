import numpy as np

from smallnet.layers import Linear, Tanh
from smallnet.model import Sequential
from smallnet.optimiser import SGD

# training data
inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [0],
    [1],
    [1],
    [0]
])

network = Sequential([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=1)
])

network.compile(optimizer=SGD(learning_rate=0.05))
network.train(inputs, targets, epochs=2000)

for x, y in zip(inputs, targets):
    predicted = network.forward(x)
    print(x, predicted, y)
