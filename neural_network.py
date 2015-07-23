class NeuralNetwork(object):

    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        cur_input = x
        for layer in self.layers:
            cur_input = layer.forward(cur_input)
        return cur_input

    def forward_backward(self, x, y):
        output = self.predict(x)

        init_delta = output - y
        cur_delta = init_delta
        for layer in reversed(self.layers):
            cur_delta = layer.backward(cur_delta)

    def get_weighted_layers(self):
        return filter(lambda l: hasattr(l, 'weights'), self.layers)
