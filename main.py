from neural_network import NeuralNetwork
from layers import LinearLayer
from layers import ActivationLayer

from solvers import SGDSolver


def main():
    network = NeuralNetwork([
        LinearLayer(4, 2),
        ActivationLayer('sigmoid'),
        # LinearLayer(8, 4),
        # ActivationLayer('sigmoid'),
        LinearLayer(2, 2),
        ActivationLayer('tanh'),
    ])

    solver = SGDSolver(network, eta=0.1)

    X = [[1, 2, 3, 4]] * 10000
    Y = [[1, -1.]] * 10000

    print network.predict([1, 2, 3, 4])
    solver.solve(X, Y)
    print network.predict([1, 2, 3, 4])


if __name__ == '__main__':
    main()
