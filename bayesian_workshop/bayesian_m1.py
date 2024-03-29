import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()  # Nice plot aesthetic
np.random.seed(101767)


class bayes_framework:
    def __init__(self):
        self.model = """
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
        self.data = dict()
        self.sm = None  # stan model
        self.fit = None

    def simulate_data(self):
        # Data Generation
        # Parameters to be inferred
        alpha = 4.0
        beta = 0.5
        sigma = 1.0

        # Generate and plot data
        x = 10 * np.random.rand(100)
        y = alpha + beta * x
        y = np.random.normal(y, scale=sigma)
        # Put our data in a dictionary
        self.data = {'N': len(x), 'x': x, 'y': y}

    def dump_data(self):
        print('x=', self.data['x'])
        print('y=', self.data['y'])

    def compile(self):
        # Compile the model
        self.sm = pystan.StanModel(model_code=self.model)

    def train(self):
        # Train the model and generate samples
        self.fit = self.sm.sampling(data=self.data, iter=1000, chains=4, warmup=500, thin=1, seed=101)

    def results(self):
        summary_dict = self.fit.summary()
        df = pd.DataFrame(summary_dict['summary'],
                          columns=summary_dict['summary_colnames'],
                          index=summary_dict['summary_rownames'])

        alpha_mean, beta_mean = df['mean']['alpha'], df['mean']['beta']

        # Extracting traces
        alpha = self.fit['alpha']
        beta = self.fit['beta']
        sigma = self.fit['sigma']
        lp = self.fit['lp__']

        print('alpha: ', len(alpha), alpha)
        print('beta: ', len(beta), beta)
        print('sigma: ', len(sigma), sigma)
        print('lp: ', len(lp), lp)

        print(df)
        print(self.fit)

    def plot_trace(self, param, param_name='parameter'):
        """Plot the trace and posterior of a parameter."""

        # Summary statistics
        mean = np.mean(param)
        median = np.median(param)
        cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

        # Plotting
        plt.subplot(2,1,1)
        plt.plot(param)
        plt.xlabel('samples')
        plt.ylabel(param_name)
        plt.axhline(mean, color='r', lw=2, linestyle='--')
        plt.axhline(median, color='c', lw=2, linestyle='--')
        plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
        plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
        plt.title('Trace and Posterior Distribution for {}'.format(param_name))

        plt.subplot(2,1,2)
        plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
        plt.xlabel(param_name)
        plt.ylabel('density')
        plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
        plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
        plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
        plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

        plt.gcf().tight_layout()
        plt.legend()


if __name__ == '__main__':
    bfw = bayes_framework()
    bfw.simulate_data()
    bfw.dump_data()
    bfw.compile()
    bfw.train()
    bfw.results()
