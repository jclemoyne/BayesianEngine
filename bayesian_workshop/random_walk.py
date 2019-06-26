import numpy as np
from math import sqrt
from scipy.stats import norm
import collections
from pylab import plot, show, grid, xlabel, ylabel


def toss_coin(N):
	outcome = []
	for i in range(N):
		outcome += [np.random.random_integers(0, 1)]
	print(outcome)
	freq = collections.Counter(outcome)
	print('{} distribution {:.4f}'.format(freq, freq[1]/freq[0]))


# a randow walk is a Markov Chain
def random_walk(L, init_state):
	states = []
	state = init_state
	for i in range(L):
		toss = np.random.random_integers(0, 1)
		if toss == 1:
			state += 1
		else:
			state -= 1
		states += [state]
	print(states)


def random_walk_multiple(Dim=3, L=20 , state_zero=10):
	init_state = (state_zero,)*Dim
	print('initial state:', init_state)
	states = []
	state_tuple = init_state
	gain = np.array([-1, +1])
	for i in range(L):
		trials = []
		state_list = list(state_tuple)
		for k in range(Dim):
			toss = np.random.random_integers(0, 1)
			state_list[k] += gain[toss]
		state_tuple = tuple(state_list)
		states += [state_tuple]
	print(states)


def brownian(x0, n, dt, delta, out=None):
	"""
	Generate an instance of Brownian motion (i.e. the Wiener process):

		X(t) = X(0) + N(0, delta**2 * t; 0, t)

	where N(a,b; t0, t1) is a normally distributed random variable with mean a and
	variance b.  The parameters t0 and t1 make explicit the statistical
	independence of N on different time intervals; that is, if [t0, t1) and
	[t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
	are independent.

	Written as an iteration scheme,

		X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


	If `x0` is an array (or array-like), each value in `x0` is treated as
	an initial condition, and the value returned is a numpy array with one
	more dimension than `x0`.

	Arguments
	---------
	x0 : float or numpy array (or something that can be converted to a numpy array
		 using numpy.asarray(x0)).
		The initial condition(s) (i.e. position(s)) of the Brownian motion.
	n : int
		The number of steps to take.
	dt : float
		The time step.
	delta : float
		delta determines the "speed" of the Brownian motion.  The random variable
		of the position at time t, X(t), has a normal distribution whose mean is
		the position at time t=0 and whose variance is delta**2*t.
	out : numpy array or None
		If `out` is not None, it specifies the array in which to put the
		result.  If `out` is None, a new numpy array is created and returned.

	Returns
	-------
	A numpy array of floats with shape `x0.shape + (n,)`.

	Note that the initial value `x0` is not included in the returned array.
	"""

	x0 = np.asarray(x0)

	# For each element of x0, generate a sample of n numbers from a
	# normal distribution.
	r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

	# If `out` was not given, create an output array.
	if out is None:
		out = np.empty(r.shape)

	# This computes the Brownian motion by forming the cumulative sum of
	# the random samples.
	np.cumsum(r, axis=-1, out=out)

	# Add the initial condition.
	out += np.expand_dims(x0, axis=-1)

	return out


def brownian_motion_example():
	# The Wiener process parameter.
	delta = 2
	# Total time.
	T = 10.0
	# Number of steps.
	N = 500
	# Time step size
	dt = T/N
	# Number of realizations to generate.
	m = 20
	# Create an empty array to store the realizations.
	x = np.empty((m, N+1))
	# Initial values of x.
	x[:, 0] = 50

	brownian(x[:,0], N, dt, delta, out=x[:, 1:])

	t = np.linspace(0.0, N*dt, N+1)
	for k in range(m):
		plot(t, x[k])
	xlabel('t', fontsize=16)
	ylabel('x', fontsize=16)
	grid(True)
	show()


if __name__ == '__main__':
	# toss_coin(1000)
	# random_walk(50, 10)
	# random_walk_multiple()
	brownian_motion_example()