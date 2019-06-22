import pystan
import numpy as np


def school_example():
	schools_dat = {'J': 8, 'y': [28, 8, -3, 7, -1, 1, 18, 12],
				'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}

	sm = pystan.StanModel(file='8schools.stan')
	fit = sm.sampling(data=schools_dat, iter=1000, chains=4)

	la = fit.extract(permuted=True)  # return a dictionary of arrays
	mu = la['mu']

	# return an array of three dimensions: iterations, chains, parameters
	a = fit.extract(permuted=False)

	print(fit)
	fit.plot()


def simulate_arma11_data():
	mu = -1.25
	sigma = 0.75
	theta = 0.5
	phi = 0.2
	T = 1000
	err = np.random.normal(0, sigma, T)
	y = np.zeros(T)
	y[0] = err[1] + mu + phi * mu
	for t in range(1, T):
		y[t] = err[t] + (mu + phi * y[t-1] + theta * err[t-1])
	print(y)
	pass


def main():
	# school_example()
	simulate_arma11_data()
	pass


if __name__ == '__main__':
	main()
