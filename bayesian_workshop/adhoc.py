from pymc.examples import disaster_model
from pymc import MCMC
from pylab import hist, show
from pymc.Matplot import plot


def show_model_elements():
    print(disaster_model.switchpoint.parents)
    print(disaster_model.disasters.parents)
    print(disaster_model.rate.children)
    print(disaster_model.disasters.value)
    print('switchpoint value: ', disaster_model.switchpoint.value)
    print('early_mean: ', disaster_model.early_mean.value)
    print('late_mean: ', disaster_model.late_mean.value)
    print('rate: ', disaster_model.rate.value)
    print('switchpoint logp: ', disaster_model.switchpoint.logp)
    print('diasters logp:', disaster_model.disasters.logp)
    print('early mean logp: ', disaster_model.early_mean.logp)
    print('late mean logp: ', disaster_model.late_mean.logp)


def fit_model():
    M = MCMC(disaster_model)
    M.sample(iter=10000, burn=1000, thin=10)
    print('switchpoint: ', M.trace('switchpoint')[:])
    print('hist: ', hist(M.trace('late_mean')[:]))
    # show()
    plot(M)


if __name__ == '__main__':
    show_model_elements()
    fit_model()
