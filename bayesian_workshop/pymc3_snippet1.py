"""
	Bayesian Inference Workshop - Anaplan San Francisco July 2019
	by Jean Claude Lemoyne
"""

import pandas as pd
import pymc3 as pm
from matplotlib import pyplot as plt


def trial1():
	radon = pd.read_csv('data/radon.csv')[['county', 'floor', 'log_radon']]
	# print(radon.head())
	county = pd.Categorical(radon['county']).codes
	# print(county)

	niter = 1000
	with pm.Model() as hm:
		# County hyperpriors
		mu_a = pm.Normal('mu_a', mu=0, sd=10)
		sigma_a = pm.HalfCauchy('sigma_a', beta=1)
		mu_b = pm.Normal('mu_b', mu=0, sd=10)
		sigma_b = pm.HalfCauchy('sigma_b', beta=1)

		# County slopes and intercepts
		a = pm.Normal('slope', mu=mu_a, sd=sigma_a, shape=len(set(county)))
		b = pm.Normal('intercept', mu=mu_b, sd=sigma_b, shape=len(set(county)))

		# Houseehold errors
		sigma = pm.Gamma("sigma", alpha=10, beta=1)

		# Model prediction of radon level
		mu = a[county] + b[county] * radon.floor.values

		# Data likelihood
		y = pm.Normal('y', mu=mu, sd=sigma, observed=radon.log_radon)

		start = pm.find_MAP()
		step = pm.NUTS(scaling=start)
		hm_trace = pm.sample(niter, step, start=start)

		plt.figure(figsize=(8, 60))
		pm.forestplot(hm_trace, varnames=['slope', 'intercept'])


if __name__ == '__main__':
	trial1()