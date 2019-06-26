"""
	Bayesian Inference Workshop - Anaplan San Francisco July 2019
	by Jean Claude Lemoyne
"""

import pandas as pd
import pymc3 as pm


def trial1():
	radon = pd.read_csv('data/radon.csv')[['county', 'floor', 'log_radon']]
	# print(radon.head())
	county = pd.Categorical(radon['county']).codes
	# print(county)

	with pm.Model() as hm:
		# County hyperpriors
		mu_a = pm.Normal('mu_a', mu=0, tau=1.0/100**2)
		sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)
		mu_b = pm.Normal('mu_b', mu=0, tau=1.0/100**2)
		sigma_b = pm.Uniform('sigma_b', lower=0, upper=100)

		# County slopes and intercepts
		a = pm.Normal('slope', mu=mu_a, sd=sigma_a, shape=len(set(county)))
		b = pm.Normal('intercept', mu=mu_b, tau=1.0/sigma_b**2, shape=len(set(county)))

		# Houseehold errors
		sigma = pm.Gamma("sigma", alpha=10, beta=1)

		# Model prediction of radon level
		mu = a[county] + b[county] * radon.floor.values

		# Data likelihood
		y = pm.Normal('y', mu=mu, sd=sigma, observed=radon.log_radon)
		print(y)


if __name__ == '__main__':
	trial1()