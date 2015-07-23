import numpy as np

from neural_network import NeuralNetwork
from layers import LinearLayer
from layers import ActivationLayer

def main():
    nn = NeuralNetwork([
        LinearLayer(4, 2),
        # ActivationLayer('sigmoid'),
    ], eta=0.01)

    print nn.predict([1,2,3,4])
    for i in range(10000):
        nn.fit(np.array([1,2,3,4]), [3.873, -4.2])
    print nn.predict([1,2,3,4])


if __name__ == '__main__':
    main()
