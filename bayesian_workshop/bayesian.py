import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import poisson

from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from functools import partial

plt.style.use('ggplot')


def first_pass():
    print('...go')
    n = 100
    h = 61
    p = h/n
    rv = st.binom(n, p)
    mu = rv.mean()

    a, b = 10, 10
    prior = st.beta(a, b)
    post = st.beta(h+a, n-h+b)
    ci = post.interval(0.95)

    thetas = np.linspace(0, 1, 200)
    plt.figure(figsize=(12, 9))
    plt.style.use('ggplot')
    plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
    plt.plot(thetas, post.pdf(thetas), label='Posterior', c='red')
    plt.plot(thetas, n*st.binom(n, thetas).pmf(h), label='Likelihood', c='green')
    plt.axvline((h+a-1)/(n+a+b-2), c='red', linestyle='dashed', alpha=0.4, label='MAP')
    plt.axvline(mu/n, c='green', linestyle='dashed', alpha=0.4, label='MLE')
    plt.xlim([0, 1])
    plt.axhline(0.3, ci[0], ci[1], c='black', linewidth=2, label='95% CI');
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.legend()
    plt.show()

    thetas = np.linspace(0, 1, 200)
    prior = st.beta(a, b)

    post = prior.pdf(thetas) * st.binom(n, thetas).pmf(h)
    post /= (post.sum() / len(thetas))

    plt.figure(figsize=(12, 9))
    plt.plot(thetas, prior.pdf(thetas), label='Prior', c='blue')
    plt.plot(thetas, n*st.binom(n, thetas).pmf(h), label='Likelihood', c='green')
    plt.plot(thetas, post, label='Posterior', c='red')
    plt.xlim([0, 1])
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('Density', fontsize=16)
    plt.legend()
    plt.show()


def target(lik, prior, n, h, theta):
    if theta < 0 or theta > 1:
        return 0
    else:
        return lik(n, theta).pmf(h)*prior.pdf(theta)


def sampler():
    n = 100
    h = 61
    a = 10
    b = 10
    lik = st.binom
    prior = st.beta(a, b)
    sigma = 0.3

    naccept = 0
    theta = 0.1
    niters = 10000
    samples = np.zeros(niters+1)
    samples[0] = theta
    for i in range(niters):
        theta_p = theta + st.norm(0, sigma).rvs()
        rho = min(1, target(lik, prior, n, h, theta_p)/target(lik, prior, n, h, theta ))
        u = np.random.uniform()
        if u < rho:
            naccept += 1
            theta = theta_p
        samples[i+1] = theta
    nmcmc = len(samples)//2
    print('len samples: ', len(samples))
    print('nmcmc: ', nmcmc)
    print("Efficiency = ", naccept/niters)

    post = st.beta(h+a, n-h+b)

    thetas = np.linspace(0, 1, 200)
    plt.figure(figsize=(12, 9))
    plt.hist(samples[nmcmc:], 40, histtype='step', density=True, linewidth=1, label='Distribution of prior samples');
    plt.hist(prior.rvs(nmcmc), 40, histtype='step', density=True, linewidth=1, label='Distribution of posterior samples');
    plt.plot(thetas, post.pdf(thetas), c='red', linestyle='--', alpha=0.5, label='True posterior')
    plt.xlim([0,1]);
    plt.legend(loc='best')
    plt.show()


def poisson_prob():
    mus = [2.5, 4, 6]
    for mu in mus:
        print('=== assume sales at a rate of {} units per week'.format(mu))
        for x in range(10):
            prob = poisson.pmf(x, mu)
            print('\tThe probability for selling ', x, ' is ', np.round(prob, 2))


if __name__ == '__main__':
    # first_pass()
    # sampler()
    poisson_prob()

