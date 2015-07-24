import random

import numpy as np


class Solver(object):

    """Solvers optimize weights in a network."""

    def __init__(self, network, eta=0.01):
        self.network = network
        self.eta = eta

    def solve(self, data, batch_size=10):
        raise NotImplementedError()

    def step(self, data):
        raise NotImplementedError()


class SGDSolver(Solver):

    """Optimizes network weights using stochastic gradient descent."""

    def solve(self, data, batch_size=10):
        """Run SGD on given training examples with specified batch size.

        Parameters
        ----------
        data: list
            each element is a tuple (x,y) pair
        batch_size: int
            number of examples to use per update

        """
        random.shuffle(data)
        batches = [data[k:k + batch_size]
                     for k in range(0, len(data), batch_size)]

        for batch in batches:
            self.step(batch)

    def step(self, data):
        """Update network parameters with given training examples.

        Parameters
        ----------
        data: list
            each element is a tuple (x,y) pair

        """
        weighted_layers = self.network.get_weighted_layers()

        nabla_w = [np.zeros(layer.weights.shape) for layer in weighted_layers]
        nabla_b = [np.zeros(layer.biases.shape) for layer in weighted_layers]

        for x, y in data:
            output = self.network.predict(x)
            self.network.forward_backward(x, y, output - y)  # `output - y` is delta for MSE
            for i, layer in enumerate(weighted_layers):
                dw, db = layer.get_gradients()
                nabla_w[i] += dw
                nabla_b[i] += db

        for i, layer in enumerate(weighted_layers):
            layer.update(
                -self.eta * (nabla_w[i] / len(data)),  # change in weights
                -self.eta * (nabla_b[i] / len(data))   # change in biases
            )

    def step_once(self, x, y):
        """Update network parameters with single example.

        Parameters
        ----------
        x: list
            single training example input
        y: list
            single training example output

        """
        self.network.forward_backward(x, y)
        for layer in self.network.get_weighted_layers():
            layer.biases -= self.eta * (layer.db / 1)
            layer.weights -= self.eta * (layer.dw / 1)
