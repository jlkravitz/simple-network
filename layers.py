import numpy as np


class Layer(object):

    """A layer in a network."""

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, delta):
        raise NotImplementedError()


class LinearLayer(Layer):

    """Linear layer of network.

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

    def get_gradients(self):
        """Return weight and bias gradients."""
        return self.dw, self.db

    def update(self, change_w, change_b):
        """Update layer weights by given amounts."""
        self.weights += change_w
        self.biases += change_b


class ActivationLayer(Layer):

    """Activation layer of network.

    Simply applies an activation function to its input.

    """

    def __init__(self, activation_type):
        if activation_type == 'sigmoid':
            self.theta = ActivationLayer.sigmoid
            self.theta_prime = ActivationLayer.sigmoid_prime
        elif activation_type == 'tanh':
            self.theta = ActivationLayer.tanh
            self.theta_prime = ActivationLayer.tanh_prime
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

    @staticmethod
    @np.vectorize
    def tanh(z):
        """The tanh function."""
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    @staticmethod
    @np.vectorize
    def tanh_prime(z):
        """The derivative of the tanh function."""
        return 1 - ActivationLayer.tanh(z)**2

    def forward(self, x):
        """Feed input forward through layer and return output."""
        self.x = x
        self.output = self.theta(x)
        return self.output

    def backward(self, output_delta):
        """Backward propagate delta and return new delta."""
        return np.multiply(output_delta, self.theta_prime(self.output))
