"""
	Bayesian Inference Workshop - Anaplan San Francisco July 2019
	by Jean Claude Lemoyne
"""
import pandas as pd
import pystan


def main():
	df = pd.read_csv('data/HtWt.csv')
	print(df.head())

	log_reg_code = """
	data {
		int<lower=0> n;
		int male[n];
		real weight[n];
		real height[n];
	}
	transformed data {}
	parameters {
		real a;
		real b;
		real c;
	}
	transformed parameters {}
	model {
		a ~ normal(0, 10);
		b ~ normal(0, 10);
		c ~ normal(0, 10);
		for(i in 1:n) {
			male[i] ~ bernoulli(inv_logit(a*weight[i] + b*height[i] + c));
	  }
	}
	generated quantities {}
	"""

	log_reg_dat = {
		'n': len(df),
		'male': df.male,
		'height': df.height,
		'weight': df.weight
	}

	fit = pystan.stan(model_code=log_reg_code, data=log_reg_dat, iter=2000, chains=1)

	print(fit)


if __name__ == '__main__':
	main()