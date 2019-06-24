import pystan
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set()  # Nice plot aesthetic
np.random.seed(101)


def simulate_data():
	# Parameters to be inferred
	alpha = 4.0
	beta = 0.5
	sigma = 1.0

	# Generate and plot data
	x = 10 * np.random.rand(100)
	y = alpha + beta * x
	y = np.random.normal(y, scale=sigma)

	return y, x


def train_model():
	model = """
data {
	int<lower=0> N;
	vector[N] x;
	vector[N] y;
}
parameters {
	real alpha;
	real beta;
	real<lower=0> sigma;
}
model {
	y ~ normal(alpha + beta * x, sigma);
}
"""
	y, x = simulate_data()
	data = {'N': len(x), 'x': x, 'y': y}
	# compile model
	sm = pystan.StanModel(model_code=model)
	fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)
	print(fit)


if __name__ == '__main__':
	train_model()