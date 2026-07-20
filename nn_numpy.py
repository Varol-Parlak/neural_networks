import numpy as np

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weight) + self.biases

    def backward(self, dvalues): # dvalues = a neurons error amount
        self.dweight = np.dot(self.inputs.T, dvalues)        
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weight.T) # dinputs = the dvalues of the previous layer 


class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs) 

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax: # gives predictions based on classes
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = (exp_values / np.sum(exp_values, axis=1, keepdims=True))

class Loss: # the entire loss of the predictions
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss): # how wrong are the predictions from softmax
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_cliped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_cliped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_cliped * y_true, axis=1)

        neglog = -np.log(correct_confidences)
        return neglog

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.softmax_activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    
    def forward(self, inputs, y_true):
        self.softmax_activation.forward(inputs)
        self.output = self.softmax_activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy() 
        self.dinputs[range(samples), y_true] -= 1 # only the true value will be -=1
        self.dinputs = self.dinputs / samples # average the gradient to not overshoot training 