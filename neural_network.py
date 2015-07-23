class NeuralNetwork(object):

    """Encapsulates a network."""

    def __init__(self, layers):
        self.layers = layers

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

    def forward_backward(self, x, y):
        """Feed input through network and back propagate error.

        Parameters
        ----------
        x: list
            input to network
        y: list
            expected output of given input `x`

        """
        output = self.predict(x)

        init_delta = output - y
        cur_delta = init_delta
        for layer in reversed(self.layers):
            cur_delta = layer.backward(cur_delta)

    def get_weighted_layers(self):
        """Return all layers which have weights.

        This method is used when updating weights in the Solver class.

        """
        return filter(lambda l: hasattr(l, 'weights'), self.layers)
