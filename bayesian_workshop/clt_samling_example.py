"""
	Knowledge Machine (c) 2019 Jean Claude Lemoyne

	This program illustrates the Central Limit Theorem (CLT)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

source = 'data/singapore-residents-by-age-group-ethnic-group-and-sex-end-june-annual.csv'


def load_data():
	df = pd.read_csv(source)
	print(df.head())
	print(df.shape)
	seq = []
	for index, row in df.iterrows():
		level_1 = row['level_1']
		if level_1.lower().find('female') > -1:
			seq += ['F']
		elif level_1.lower().find('male') > -1:
			seq += ['M']
	return seq


def sampler(dist, seq, percent=0.1):
	"""
		TO DO sample using the dist process
		return male ratio and female ratio as n_male, n_female over total gender
	"""
	pass


def resample(dist, niter):
	"""
		run sampler niter times and calculate male ratios average
		plot history
	"""
	pass

"""
	Cauchy pdf is given by f(x) = 1 / pi * (1 + x^2)
"""
if __name__ == '__main__':
	resample(np.random.uniform, 1000)
	resample(np.random.standard_cauchy, 1000)
	# resample(np.random.laplace, 1000)
	# resample(np.random.lognormal, 1000)
	# resample(np.random.logistic, 1000)
	# resample(np.random.poisson, 1000)
	plt.show()


