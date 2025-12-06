class Loss:
    pass    

class Loss_MSE(Loss):
    def forward(self, output, y):
        sample_losses = np.mean((y - output) ** 2, axis=-1)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        outputs = len(y_pred[0])
        self.dinputs = -2 * (y_true - y_pred) / outputs
        self.dinputs = self.dinputs / samples
        return self.dinputs
