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
	n = 0
	n_male = 0
	n_female = 0
	for G in seq:
		n += 1
		# u = np.random.uniform()
		u = dist()
		if u > percent:
			continue
		if G == 'F':
			n_female += 1
		elif G == 'M':
			n_male += 1

	total_gender = n_male + n_female
	male_ratio = float(n_male) / float(total_gender)
	female_ratio = float(n_female) / float(total_gender)
	# print ('# male: {} {:.4f}'.format(n_male, male_ratio))
	# print ('# female: {} {:.4f}'.format(n_female, female_ratio))
	return male_ratio, female_ratio


def resample(dist, niter):
	seq = load_data()
	nbins = int(niter / 10)
	print('# bin: ', nbins)
	mratios = []
	for i in range(niter):
		mr, fr = sampler(dist, seq)
		mratios += [mr]
		j = i + 1
		if j % 5 == 0:
			if j % 100 == 0:
				print('.')
			else:
				print('.', end = '')
	print()
	# print(mratios)
	plt.hist(mratios, density=True, bins=nbins)
	plt.ylabel('male ratio')


if __name__ == '__main__':
	resample(np.random.uniform, 1000)
	# resample(np.random.standard_cauchy, 1000)
	plt.show()
