import numpy as np


class NeuralNetwork(object):

    def __init__(self, layers, eta):
        self.eta = eta
        self.layers = layers

    def fit(self, x, y):
        cur_input = x
        for layer in self.layers:
            cur_input = layer.forward(cur_input)

        init_delta = self.layers[-1].output - y
        cur_delta = init_delta
        for layer in reversed(self.layers):
            cur_delta = layer.backward(cur_delta)

        for layer in filter(lambda l: hasattr(l, 'weights'), self.layers):
            layer.biases -= self.eta * (layer.db / 1)
            layer.weights -= self.eta * (layer.dw / 1)


    def predict(self, x):
        cur_input = x
        for layer in self.layers:
            cur_input = layer.forward(cur_input)
        return cur_input
