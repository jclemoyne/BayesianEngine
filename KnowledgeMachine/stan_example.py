import pystan
# import os
# os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

def main():
	schools_code = """
	data {
		int<lower=0> J; // number of schools
		vector[J] y; // estimated treatment effects
		vector<lower=0>[J] sigma; // s.e. of effect estimates
	}
	parameters {
		real mu;
		real<lower=0> tau;
		vector[J] eta;
	}
	transformed parameters {
		vector[J] theta;
		theta = mu + tau * eta;
	}
	model {
		eta ~ normal(0, 1);
		y ~ normal(theta, sigma);
	}
	"""

	schools_dat = {'J': 8,
				   'y': [28,  8, -3,  7, -1,  1, 18, 12],
				   'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

	sm = pystan.StanModel(model_code=schools_code)
	fit = sm.sampling(data=schools_dat, algorithm='HMC', iter=1000, chains=4)
	print(fit)

if __name__ == '__main__':
	main()