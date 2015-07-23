import numpy as np


class Layer(object):

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, delta):
        raise NotImplementedError()


class LinearLayer(Layer):
    """Linear layer of neural network.

    Performs weighted element-wise addition.

    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size))
        self.biases = np.zeros(output_size)

    def forward(self, x):
        """Feed input forward through layer and return output."""
        self.x = x
        self.output = self.weights.dot(x) + self.biases
        return self.output

    def backward(self, output_delta):
        """Backward propagate delta and return new delta."""
        self.db = output_delta
        self.dw = np.outer(output_delta, self.x)
        return self.weights.T.dot(output_delta)  # layer input deltas


class ActivationLayer(Layer):
    def __init__(self, activation_type):
        if activation_type == 'sigmoid':
            self.theta = ActivationLayer.sigmoid
            self.theta_prime = ActivationLayer.sigmoid_prime
        else:
            raise ValueError('Given activation is not available.')

    @staticmethod
    @np.vectorize
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    @np.vectorize
    def sigmoid_prime(z):
        """The derivative of the sigmoid function."""
        return ActivationLayer.sigmoid(z) - ActivationLayer.sigmoid(z)**2

    def forward(self, x):
        """Feed input forward through layer and return output."""
        self.x = x
        self.output = self.theta(x)
        return self.output

    def backward(self, output_delta):
        """Backward propagate delta and return new delta."""
        return np.multiply(output_delta, self.theta_prime(self.output))
