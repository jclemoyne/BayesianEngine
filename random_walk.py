import numpy as np
import collections


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


if __name__ == '__main__':
	# toss_coin(1000)
	# random_walk(50, 10)
	random_walk_multiple()