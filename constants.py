import numpy as np

class Phases(object):
    """Constants for training and testing phases."""
    TRAIN = 0
    TEST = 1

class ActivationFunctions(object):
    """Definitions of activation functions and their derivatives."""
    sigmoid = np.vectorize(lambda z: 1.0 / (1.0 + np.exp(-z)))
    sigmoid_prime = np.vectorize(lambda sigmoid_out: sigmoid_out - sigmoid_out**2)
    tanh = np.vectorize(lambda z: (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1))
    tanh_prime = np.vectorize(lambda tanh_out: 1 - tanh_out**2)
    relu = np.vectorize(lambda z: max(0, z))
    relu_prime = np.vectorize(lambda relu_out: int(relu_out > 0))
