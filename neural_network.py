import constants as C


class NeuralNetwork(object):

    """Encapsulates a network."""

    def __init__(self, layers, phase=C.Phases.TEST):
        self.layers = layers
        self.set_phase(phase)

    def set_phase(self, phase):
        """Set phase for network and its layers."""
        self.phase = phase
        for layer in self.layers:
            layer.set_phase(phase)

    def predict(self, x):
        """Feed the input through the network and return the result.

        Parameters
        ----------
        x: list
            input to network

        """
        cur_input = x
        for layer in self.layers:
            cur_input = layer.forward(cur_input)
        return cur_input

    def forward_backward(self, x, y, output_delta):
        """Feed input through network and back propagate error.

        Parameters
        ----------
        x: list
            input to network
        y: list
            expected output of given input `x`
        output_delta: float
            delta of output layer; since this depends on the loss function,
            we let the solver pass the output delta to this method.

        """
        cur_delta = output_delta
        for layer in reversed(self.layers):
            cur_delta = layer.backward(cur_delta)

    def get_weighted_layers(self):
        """Return all layers which have weights.

        This method is used when updating weights in the Solver class.

        """
        return filter(lambda l: hasattr(l, 'weights'), self.layers)
