import os
import struct
import numpy as np

class Optimizer_Adam:   
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache_m'):
            layer.weight_cache_m = np.zeros_like(layer.weight)
            layer.weight_cache_v = np.zeros_like(layer.weight)
            layer.bias_cache_m = np.zeros_like(layer.biases)
            layer.bias_cache_v = np.zeros_like(layer.biases)
        
        if layer == dense1: 
            self.iterations += 1
        t = self.iterations

        layer.weight_cache_m = self.beta_1 * layer.weight_cache_m + (1 - self.beta_1) * layer.dweight
        layer.weight_cache_v = self.beta_2 * layer.weight_cache_v + (1 - self.beta_2) * (layer.dweight ** 2)
        
        m_w_corrected = layer.weight_cache_m / (1 - self.beta_1 ** t)
        v_w_corrected = layer.weight_cache_v / (1 - self.beta_2 ** t)
        
        layer.weight += -self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)

        layer.bias_cache_m = self.beta_1 * layer.bias_cache_m + (1 - self.beta_1) * layer.dbiases
        layer.bias_cache_v = self.beta_2 * layer.bias_cache_v + (1 - self.beta_2) * (layer.dbiases ** 2)
        
        m_b_corrected = layer.bias_cache_m / (1 - self.beta_1 ** t)
        v_b_corrected = layer.bias_cache_v / (1 - self.beta_2 ** t)
        
        layer.biases += -self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weight) + self.biases

    def backward(self, dvalues): # dvalues = a neurons error amount (d = derivative)
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

# Loading the dataset
# ------------------------------------
def load_mnist_idx(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        
    return images, labels

DATA_DIR = "mnist_dataset"
train_img_path = os.path.join(DATA_DIR, "train-images-idx3-ubyte")
train_lbl_path = os.path.join(DATA_DIR, "train-labels-idx1-ubyte")
test_img_path = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte")
test_lbl_path = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte")
X_train_raw, y_train = load_mnist_idx(train_img_path, train_lbl_path)
X_test_raw, y_test = load_mnist_idx(test_img_path, test_lbl_path)
X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0
# ------------------------------------

