import numpy as np
import math
from scipy import signal as sg

eta=0.01
batch_size=1
#INPUT format-----
# d -dimension of input
# m - number of input data
# X - (d, m)

#Yet to implement
#RMSprop
#GD with momentum
#Adam optimization
#Conv layer
#Pooling layer


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def activation_func(arr, basis_func):
	if basis_func == "tanh":
		a = np.tanh(arr)
		return a
	elif basis_func == "sigmoid":
		a = sigmoid(arr)
		return a
	elif basis_func == "relu":
		a = np.maximum(arr,0)
		return a
	else:
		return arr

def act_derivative(arr, basis_func):
	if basis_func == "tanh":
		a= 1 - np.tanh(arr) * np.tanh(arr)
		return a
	elif basis_func == "sigmoid":
		a = activation_func(arr, basis_func)
		a = a * (1 - a)
		return a
	elif basis_func == "relu":
		a = (arr > 0)
		return a

def cross_entropy_cost(output, labels):
	cost = (np.dot(np.log(output), labels.T) + np.dot(np.log(1 - output),(1 - labels).T))/batch_size



class convolution_layer:
	def __init__(self, kernel_size, num_kernel):
		self.num_kernel=num_kernel
		self.kernel_size=kernel_size
		self.kernels=[np.random.rand(kernel_size, kernel_size) for i in range(0,num_kernel)]
	def compute_convolution(self):
		res = sg.convolve(image, self.kernel, "valid")
	def show_kernel(self):
		for i in range(0,num_kernel):
			print np.shape(self.kernel)
	def update_weights(self, update):
		self.kernels=update

class pooling_layer:
	def __init__(self, pool_method, image):
		self.pool_method = pool_method
		self.input_size = input_size
		self.image = image

class fully_connected_layer:
	def __init__(self, num_input_units, num_units, basis_func):
		self.num_inputs_units = num_input_units
		self.basis_func = basis_func
		#FIX this Option for weight initialization - Xaviers/He
		# Weights of shape (n[l-1], n[l])
		self.layer_weights = np.random.randn(num_units, num_input_units) * 0.01
		self.layer_bias = np.zeros((num_units, 1))
		self.before_activation = np.zeros((num_units,1))
		self.activations = np.zeros((num_units,1))
		self.weight_delta = np.zeros((num_units, num_input_units))
		self.bias_delta = np.zeros((num_units, 1))
		self.der_wrt_layer_input = np.zeros((num_input_units, batch_size))
	def forward_prop(self, input):
		self.before_activation =  np.dot(self.layer_weights, input) + self.layer_bias
		#print self.before_activation
		self.activations = activation_func(self.before_activation, self.basis_func)
		return self.activations
        ''' TODO batch size option '''
	def update_weights(self, input, err, labels):
                batch_size = input.shape[1]
		if self.basis_func == "linear":
			delta = self.activations - labels
			self.der_wrt_layer_input = np.dot((self.layer_weights).T, delta)
			self.weight_delta = np.dot(delta, input.T)/batch_size
			self.bias_delta = np.sum(delta, axis = 1, keepdims = True)/batch_size
			self.layer_weights = self.layer_weights - eta * self.weight_delta
			self.layer_bias = self.layer_bias - eta * self.bias_delta
		else:
		#Verify below
			delta = err * act_derivative(self.before_activation, self.basis_func)
			self.der_wrt_layer_input = np.dot((self.layer_weights).T, delta)
			self.weight_delta = np.dot(delta, input.T)/batch_size
			self.bias_delta = np.sum(delta, axis = 1, keepdims=True)/batch_size
			self.layer_weights = self.layer_weights - eta * self.weight_delta
			self.layer_bias = self.layer_bias - eta * self.bias_delta
		return self.der_wrt_layer_input
	def get_weights(self):
		return self.layer_weights

class nn_model:
    def __init__(self, layers, learning_rate, num_epochs, batch_size):
        self.learning_rate = learning_rate
        self.layers = layers
        self.num_layers = len(layers)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    def train_network(self, train_x, train_y):
        for e in range(self.num_epochs):
            ''' BATCH SIZE WORKS???'''
            num_mini_batches = train_x.shape[1]/self.batch_size
            for j in range(num_mini_batches):
                mini_batch_x = train_x[:, j * self.batch_size : (j + 1) * self.batch_size]
                mini_batch_y = train_y[:, j * self.batch_size : (j + 1) * self.batch_size]
                ''' Forward Propagation '''
                forward_cache = {"Z0": mini_batch_x}
                for i in range(self.num_layers):
                    forward_cache["Z" + str(i + 1)] = self.layers["L" + str(i + 1)].forward_prop(forward_cache["Z" + str(i)])
                ''' Back Propagation '''
                temp_err = np.zeros((1,1))
                for i in range(self.num_layers, 0, -1):
                   temp_err = self.layers["L" + str(i)].update_weights(forward_cache["Z" + str(i - 1)], temp_err, mini_batch_y)
    def predict(self, test_x):
        layer_output = test_x
        for i in range(self.num_layers):
            layer_output = self.layers["L" + str(i + 1)].forward_prop(layer_output)
        return layer_output
