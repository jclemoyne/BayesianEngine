import numpy as np
from scipy.stats import logistic

verbose = True

X = np.array([	[1.0, 1.0, 0.0, 0.0, 1.0],
				[0.0, 0.0, 1.0, 1.0, 0.0],
				[1.0, 0.0, 1.0, 1.0, 0.0],
				[1.0, 1.0, 1.0, 0.0, 1.0],
				[1.0, 1.0, 0.0, 0.0, 1.0],
				[0.0, 1.0, 1.0, 1.0, 0.0],
				[0.0, 1.0, 1.0, 0.0, 0.0],
				[1.0, 0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0, 1.0],
				[1.0, 0.0, 0.0, 0.0, 1.0]])

Y = np.array([	[1.0],
				[1.0],
				[1.0],
				[0.0],
				[1.0],
				[1.0],
				[1.0],
				[0.0],
				[0.0],
				[1.0]])


def x_rand_instance(n):
	x = np.random.rand(n)
	if verbose:
		print('x = ', x)
	return x


def simulated_weights(x):
	n = len(x)
	w = np.random.randn(n)
	s = np.sum(w)
	w = w / s
	if verbose:
		print('w = ', w)
	return w


def sigmoid(x, beta=0, alpha=1):
	y = 1 / (1 + np.exp(-alpha*(x-beta)))
	return y


def sigmoid_derivative(output):
	return output * (1-output)


def evaluate_gradient(learning_rate, output):
	return learning_rate * sigmoid_derivative(output)


def activation(level, t, show):
	if show:
		print('av = ', level)

	if level > t:
		return 1
	else:
		return 0


def predictor_logistic(x, w, b, threshold, show=True):
	z = np.sum(np.dot(x, w)) + b
	y = activation(logistic.cdf(z), threshold, show=show)
	return y


def predictor_sigmoid(x, w, b, threshold, alpha, beta, show=True):
	z = np.sum(np.dot(x, w)) + b
	y = activation(sigmoid(z, alpha, beta), threshold, show=show)
	return y


def neuron(x, w, b, threshold):
	z = np.sum(np.dot(x, w)) + b
	y = predictor_logistic(x, w, b, threshold, show=True)
	print('y (logistic) = ', y)

	y = activation(sigmoid(z, -1, 2), threshold, show=True)
	print('y (sigmoid) = ', y)


def update_weights(x, w, learning_rate, expected, predicted):
	return w + learning_rate * (expected - predicted) * x


def update_bias(b, learning_rate, expected, predicted):
	return b + learning_rate * (expected - predicted)


# Using SGD (Stochastic Gradient Descent)
def perceptron():
	# initialize weights randomly with mean 0
	sample_size, dim = X.shape
	synapse_0 = 2*np.random.random((dim, 1)) - 1
	learning_rate = 1.0
	for iteration in range(10000):
		# forward propagation
		layer_0 = X
		layer_1 = sigmoid(np.dot(layer_0, synapse_0))
		if iteration == 0:
			print(layer_1)

		# how much did we miss?
		layer_1_error = layer_1 - Y

		# multiply how much we missed by the
		# slope of the sigmoid at the values in l1
		layer_1_delta = layer_1_error * evaluate_gradient(learning_rate, layer_1)
		synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

		# update weights
		synapse_0 -= synapse_0_derivative
	print("Output After Training:")
	np.set_printoptions(formatter={'float': '{: 1.3f}'.format})
	print(np.rint(layer_1))
	print(layer_1 - Y)


def dump_training_data():
	N, M = X.shape
	threshold = 0.8
	sample_size = N
	for i in range(sample_size):
		x = X[i]
		print('instance (%d): %s' % (i + 1, x))
		w = simulated_weights(x)
		bias = 0
		neuron(x, w, bias, threshold)


def main():
	# dump_training_data()
	perceptron()


if __name__ == '__main__':
	main()