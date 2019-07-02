import pandas as pd
import numpy as np

source = 'data/singapore-residents-by-age-group-ethnic-group-and-sex-end-june-annual.csv'


def samplger(percent=0.1):
	df = pd.read_csv(source)
	print(df.head())
	print(df.shape)
	n = 0
	n_male = 0
	n_female = 0
	for index, row in df.iterrows():
		level_1 = row['level_1']
		# print(level_1)
		n += 1
		u = np.random.uniform()
		if u > percent:
			continue
		if level_1.lower().find('female') > -1:
			n_female += 1
		elif level_1.lower().find('male') > -1:
			n_male += 1

	total_gender = n_male + n_female

	print ('# male: {} {:.4f}'.format(n_male, float(n_male) / float(total_gender)))
	print ('# female: {} {:.4f}'.format(n_female, float(n_female) / float(total_gender)))


if __name__ == '__main__':
	samplger()