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
        self.dinputs = dvalues * (1 - self.output ** 2)
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

class Activation_Softmax:
    def forward(self, inputs):
        # Remember input values for backward pass
        self.inputs = inputs
        # Subtract max value for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the values to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        # Create uninitialized gradient array
        self.dinputs = np.empty_like(dvalues)

        # Loop over samples
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self.dinputs


