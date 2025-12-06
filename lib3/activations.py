import numpy as np


class Activation:
    pass


class Activation_Tanh(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, dvalues):
        # Derivative of tanh: 1 - tanh(x)^2
        self.dinputs = dvalues * (1 - self.output**2)
        return self.dinputs


class Activation_Sigmoid(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        sigmoid_deriv = self.output * (1 - self.output)
        self.dinputs = dvalues * sigmoid_deriv
        return self.dinputs


class Activation_ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
