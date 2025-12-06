class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_params(self, layers):
        for layer in layers:
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases
