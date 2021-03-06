import numpy as np
import math
from scipy import signal as sg
from matplotlib import pyplot as plt

batch_size=1
beta=0.9
beta2=0.999
epsilon=1e-8
#INPUT format-----
# d -dimension of input
# m - number of input data
# X - (d, m)

#Yet to implement
#GD with momentum
#Adam optimization
#Conv layer
#Pooling layer
#Cost plot not correct

def gradient_descent(parameters, delta, vel, s, opt, eta):
    weight_delta, bias_delta = delta
    layer_weights, layer_bias = parameters
    vel_weights, vel_bias = vel
    s_weights, s_bias = s
    if opt == 'sgd':
	layer_weights = layer_weights - eta * weight_delta
	layer_bias = layer_bias - eta * bias_delta
        return layer_weights, layer_bias, vel_weights, vel_bias, s_weights, s_bias

    vel_weights = beta * vel_weights + (1 - beta) * weight_delta
    vel_bias = beta * vel_bias + (1 - beta) * bias_delta
    if opt == 'gd_with_momentum':
	layer_weights = layer_weights - eta * vel_weights
	layer_bias = layer_bias - eta * vel_bias
        return layer_weights, layer_bias, vel_weights, vel_bias, s_weights, s_bias
    elif opt == 'adam':
        ''' Need to verify '''
        v_weights_corr = vel_weights/(1 - beta**2)
        v_bias_corr = vel_bias/(1 - beta**2)
        s_weights = beta2 * s_weights + (1 - beta2) * weight_delta * weight_delta
        s_bias = beta2 * s_bias + (1 - beta2) * bias_delta * bias_delta
        s_weights_corr = s_weights/(1 - beta2**2)
        s_bias_corr = s_bias/(1 - beta2**2)
	layer_weights = layer_weights - eta * (np.divide(v_weights_corr, np.sqrt(s_weights_corr) + epsilon))
	layer_bias = layer_bias - eta * (np.divide(v_bias_corr, np.sqrt(s_bias_corr) + epsilon))
        return layer_weights, layer_bias, vel_weights, vel_bias, s_weights, s_bias

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def activation_func(arr, basis_func):
	if basis_func == "tanh":
		return np.tanh(arr)
	elif basis_func == "sigmoid":
		return sigmoid(arr)
	elif basis_func == "relu":
		return np.maximum(arr,0)
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

def softmax(arr):
    return np.exp(arr)/np.sum(np.exp(arr), axis=0)

def cross_entropy_cost(output, labels):
	cost = -(np.dot(np.log(output), labels.T) + np.dot(np.log(1 - output),(1 - labels).T))/batch_size
        return cost

def sum_of_squares_cost(output, labels):
	cost = -np.sum(np.square(output - labels))/batch_size
        return cost


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

''' RNN and LSTM unit. TODO: BACKPROP'''

class rnn_unit:
    def __init__(self, num_input, num_act, num_y):
        self.weights_ax = np.random.randn(num_act, num_input) * 0.01
        self.weights_aa = np.random.randn(num_act, num_input) * 0.01
        self.weights_ya = np.random.randn(num_y, num_act) * 0.01
        self.bias_a = np.zeros(num_act,1)
        self.bias_y = np.zeros(num_y, 1)
        self.activations = np.zeros(num_act, 1)
        self.output = np.zeros(num_y, 1)
    def forward_prop(self, act_prev, x):
        self.activations = activation_func(np.dot(self.weights_ax, x) + np.dot(self.weights_aa, act_prev) + self.bias_a, "tanh")
        self.output = softmax(np.dot(self.weights_ya, self.activations) + self.bias_y)
        return self.activations
    def update_weights(self, x, del_act_next, a_prev):
        del_tanh = (1 - self.activations * self.activations) * del_act_next
        del_x = np.dot(self.weights_ax.T, del_tanh)
        del_weigth_ax = np.dot(del_tanh, x.T)
        del_a_prev = np.dot(self.weights_aa.T, del_tanh)
        del_weigth_aa = np.dot(del_tanh, a_prev.T)
        del_bias_a = np.sum(del_tanh, axis=1, keepdims=True)

class lstm_unit:
    def __init__(self, num_input, num_act, num_y):
        self.weights_forget = np.random.randn(num_act, num_act + num_input) * 0.01
        self.weights_update = np.random.randn(num_act, num_act + num_input) * 0.01
        self.weights_output = np.random.randn(num_act, num_act + num_input) * 0.01
        self.weights_cell = np.random.randn(num_act, num_act + num_input) * 0.01
        self.weights_y = np.random.randn(num_y, num_act + num_input) * 0.01
        self.bias_forget = np.zeros(num_act,1)
        self.bias_update = np.zeros(num_act, 1)
        self.bias_output = np.zeros(num_act, 1)
        self.bias_cell = np.zeros(num_act, 1)
        self.bias_y = np.zeros(num_y, 1)
        self.activations = np.zeros(num_act, 1)
        self.cell_state = np.zeros(num_act, 1)
        self.output = np.zeros(num_y, 1)
    def forward_prop(self, c_prev, act_prev, x):
        n_x, m =  x.shape
        n_a = a.shape[0]
        concat = np.zeros((n_x + n_a, m))
        concat[: n_a, :] = act_prev
        concat[n_a :, :] = x

        f_gate = sigmoid(np.dot(self.weights_forget, concat) + self.bias_forget)
        u_gate = sigmoid(np.dot(self.weights_update, concat) + self.bias_update)
        c_gate = np.tanh(np.dot(self.weights_cell, concat) + self.bias_cell)
        self.cell_state = f_gate * c_prev + u_gate * c_gate
        o_gate = sigmoid(np.dot(self.weights_output) + self.bias_output)
        self.activations = o_gate * np.tanh(self.cell_state)
        self.output = softmax(np.dot(self.weights_y, self.activations) + self.bias_y)
        return self.activations, self.cell_state


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
                #Below used for optimization
                self.vel_weights = np.zeros((num_units, num_input_units))
                self.vel_bias = np.zeros((num_units, 1))
                self.s_weights = np.zeros((num_units, num_input_units))
                self.s_bias = np.zeros((num_units, 1))
	def forward_prop(self, input):
		self.before_activation =  np.dot(self.layer_weights, input) + self.layer_bias
		#print self.before_activation
		self.activations = activation_func(self.before_activation, self.basis_func)
		return self.activations
        ''' TODO batch size option '''
	def update_weights(self, input, err, labels, eta):
                batch_size = input.shape[1]
                g_delta = (self.weight_delta, self.bias_delta)
                parameters = (self.layer_weights, self.layer_bias)
                vel = (self.vel_weights, self.vel_bias)
                s = (self.s_weights, self.s_bias)
		if self.basis_func == "linear":
			delta = self.activations - labels
			self.der_wrt_layer_input = np.dot((self.layer_weights).T, delta)
			self.weight_delta = np.dot(delta, input.T)/batch_size
			self.bias_delta = np.sum(delta, axis = 1, keepdims = True)/batch_size
                        self.layer_weights ,self.layer_bias, self.vel_weights, self.vel_bias, self.s_weights, self.s_bias = gradient_descent(parameters, g_delta, vel, s, 'adam', eta)
		else:
		#Verify below
			delta = err * act_derivative(self.before_activation, self.basis_func)
			self.der_wrt_layer_input = np.dot((self.layer_weights).T, delta)
			self.weight_delta = np.dot(delta, input.T)/batch_size
			self.bias_delta = np.sum(delta, axis = 1, keepdims=True)/batch_size
                        self.layer_weights ,self.layer_bias, self.vel_weights, self.vel_bias, self.s_weights, self.s_bias = gradient_descent(parameters, g_delta, vel, s, 'adam', eta)
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
        self.cost = []
    def train_network(self, train_x, train_y):
        cost = 0
        for e in range(self.num_epochs):
            ''' BATCH SIZE WORKS???'''
            num_mini_batches = train_x.shape[1]/self.batch_size
            ''' Randomize mini batches ???? '''
            for j in range(num_mini_batches):
                mini_batch_x = train_x[:, j * self.batch_size : (j + 1) * self.batch_size]
                mini_batch_y = train_y[:, j * self.batch_size : (j + 1) * self.batch_size]
                ''' Forward Propagation '''
                forward_cache = {"Z0": mini_batch_x}
                for i in range(self.num_layers):
                    forward_cache["Z" + str(i + 1)] = self.layers["L" + str(i + 1)].forward_prop(forward_cache["Z" + str(i)])
                cost += sum_of_squares_cost(forward_cache["Z" + str(self.num_layers)], train_y)
                ''' Back Propagation '''
                temp_err = np.zeros((1,1))
                for i in range(self.num_layers, 0, -1):
                   temp_err = self.layers["L" + str(i)].update_weights(forward_cache["Z" + str(i - 1)], temp_err, mini_batch_y, self.learning_rate)
            (self.cost).append(cost/train_x.shape[1])
    def predict(self, test_x):
        layer_output = test_x
        for i in range(self.num_layers):
            layer_output = self.layers["L" + str(i + 1)].forward_prop(layer_output)
        return layer_output
    def cost_plot(self):
        cost = np.array(self.cost)
        cost.reshape((-1,1))
        epoch = np.array(range(self.num_epochs))
        epoch = epoch + 1
        epoch.reshape((-1,1))
        plt.plot(epoch, cost, 'r')
        plt.show()
