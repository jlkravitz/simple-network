import numpy as np

import constants as C


class Layer(object):

    """A layer in a network."""

    def __init__(self, phase=C.Phases.TRAIN):
        self.phase = phase

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, delta):
        raise NotImplementedError()

    def set_phase(self, phase):
        self.phase = phase


class LinearLayer(Layer):

    """Linear layer of network.

    Performs weighted element-wise addition.

    """

    def __init__(self, input_size, output_size):
        super(LinearLayer, self)
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
        super(ActivationLayer, self)
        if activation_type == 'sigmoid':
            self.theta = C.ActivationFunctions.sigmoid
            self.theta_prime = C.ActivationFunctions.sigmoid_prime
        elif activation_type == 'tanh':
            self.theta = C.ActivationFunctions.tanh
            self.theta_prime = C.ActivationFunctions.tanh_prime
        elif activation_type == 'relu':
            self.theta = C.ActivationFunctions.relu
            self.theta_prime = C.ActivationFunctions.relu_prime
        else:
            raise ValueError('Given activation is not available.')

    def forward(self, x):
        """Feed input forward through layer and return output."""
        self.x = x
        self.output = self.theta(x)
        return self.output

    def backward(self, output_delta):
        """Backward propagate delta and return new delta."""
        return np.multiply(output_delta, self.theta_prime(self.output))


class DropoutLayer(Layer):

    """Dropout layer of network; drops random input for the next layer."""

    def __init__(self, threshold=0.5):
        super(DropoutLayer, self)
        self.threshold = threshold

    def forward(self, x):
        if self.phase == C.Phases.TRAIN:
            self.mask = np.random.binomial(1, 1 - self.threshold, size=x.shape)
            return x * self.mask
        else:
            return (1 - self.threshold) * x

    def backward(self, output_delta):
        if self.phase == C.Phases.TRAIN:
            return output_delta * self.mask
        else:
            return output_delta
