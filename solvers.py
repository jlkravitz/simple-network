import random

import numpy as np


class Solver(object):

    """Solvers optimize weights in a network."""

    def __init__(self, network, eta=0.01):
        self.network = network
        self.eta = eta

    def solve(self, X, Y, batch_size=10):
        raise NotImplementedError()

    def step(self, X, Y):
        raise NotImplementedError()


class SGDSolver(Solver):

    """Optimizes network weights using stochastic gradient descent."""

    def solve(self, X, Y, batch_size=10):
        """Run SGD on given training examples with specified batch size.

        Parameters
        ----------
        X: list
            training example inputs
        Y: list
            training example outputs
        batch_size: int
            number of examples to use per update

        """
        random.shuffle(X)
        X_batches = [X[k:k + batch_size]
                     for k in range(0, len(X), batch_size)]
        Y_batches = [Y[k:k + batch_size]
                     for k in range(0, len(Y), batch_size)]

        for X_batch, Y_batch in zip(X_batches, Y_batches):
            # if i * batch_size % 1000 == 0:
            #     print i * batch_size
            self.step(X_batch, Y_batch)

    def step(self, X, Y):
        """Update network parameters with given inputs and corresponding outputs.

        Parameters
        ----------
        X: list
            training example inputs
        Y: list
            training example outputs

        """
        weighted_layers = self.network.get_weighted_layers()

        nabla_w = [np.zeros(layer.weights.shape) for layer in weighted_layers]
        nabla_b = [np.zeros(layer.biases.shape) for layer in weighted_layers]

        for x, y in zip(X, Y):
            self.network.forward_backward(x, y)
            for i, layer in enumerate(weighted_layers):
                dw, db = layer.get_gradients()
                nabla_w[i] += dw
                nabla_b[i] += db

        for i, layer in enumerate(weighted_layers):
            layer.update(
                -self.eta * (nabla_w[i] / len(X)),  # change in weights
                -self.eta * (nabla_b[i] / len(X))   # change in biases
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
