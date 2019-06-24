import numpy as np
from scipy.stats import logistic

verbose = True


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


def perceptron(x, w, b, threshold):
	pass


def main():
	N = 20
	threshold = 0.8
	sample_size = 10
	for i in range(sample_size):
		x = x_rand_instance(N)
		w = simulated_weights(x)
		bias = 0
		neuron(x, w, bias, threshold)


if __name__ == '__main__':
	main()
