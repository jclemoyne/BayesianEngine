import numpy as np
import pystan
import pickle

models_dir = 'stan_models/'
model_name = models_dir + 'arma11.pkl'
model_stan = models_dir + "stan/arma11.stan"

np.random.seed(101254127)


def simulate_arma11_data():
	mu = -1.25
	sigma = 0.75
	theta = 0.5
	phi = 0.2
	T = 1000
	err = np.random.normal(0.0, sigma, T)
	y = np.zeros(T)
	y[0] = err[1] + mu + phi * mu
	for t in range(1, T):
		y[t] = err[t] + (mu + phi * y[t-1] + theta * err[t-1])
	# print(y)
	return T, y


def train_arma11():
	model = """
data {
  int<lower=1> T;       // number of observations
  real y[T];            // observed outputs
}

parameters {
  real mu;              // mean term
  real phi;             // autoregression coeff
  real theta;           // moving avg coeff
  real<lower=0> sigma;  // noise scale
}
model {
  vector[T] nu;         // prediction for time t
  vector[T] err;        // error for time t
  nu[1] = mu + phi * mu;   // assume err[0] == 0
  err[1] = y[1] - nu[1];
  for (t in 2:T) {
    nu[t] = mu + phi * y[t-1] + theta * err[t-1];
    err[t] = y[t] - nu[t];
  }

  // priors
  mu ~ normal(0,10);
  phi ~ normal(0,2);
  theta ~ normal(0,2);
  sigma ~ cauchy(0,5);

  // likelihood
  err ~ normal(0,sigma);
}

// alternative encoding:
//
// model {
//   err <- 0;
//   for (t in 2:T) {
//     err <- y[t-1] - (mu + phi * y[t-1] + theta * err[t-1]);
//     err ~ normal(0,sigma);
//   }
	"""
	# sm = pystan.StanModel(model_code=model)
	sm = pystan.StanModel(file=model_stan)
	T, y = simulate_arma11_data()
	data = {'T': T, 'y': y}
	fit = sm.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101)
	print(fit)
	pickle.dump(sm, open(model_name, 'wb'))


def reuse_resample():
	sm = pickle.load(open(model_name, 'rb'))
	T, y = simulate_arma11_data()
	data = {'T': T, 'y': y}
	fit = sm.sampling(data=data, iter=5000, chains=4, warmup=500, thin=1, seed=101)
	print(fit)


if __name__ == '__main__':
	# simulate_arma11_data()
	# train_arma11()
	reuse_resample()