import numpy as np
import sklearn


def kernel (x,y,b):
	return np.exp (-b * (x-y))


def drawExpRV (param, rng):
	return rng.exponential (scale=param)


def simulate_time(alpha, omega, T, numEvents=None, checkStability=False, seed=None):
	dim = alpha.shape[0]
	# make mu small!
	mu_endo = np.random.uniform(0, 0.05, dim)
	prng = sklearn.utils.check_random_state (seed)
	nTotal = 0
	history = list ()
	# Initialization
	if numEvents is None:
		nExpected = np.iinfo (np.int32).max
	else:
		nExpected = numEvents
	s = 0.0

	if checkStability:
		w,v = np.linalg.eig (alpha)
		maxEig = np.amax (np.abs(w))
		if maxEig >= 1:
			print("(WARNING) Unstable ... max eigen value is: {0}".format (maxEig))

	Istar = np.sum(mu_endo)
	s += drawExpRV (1. / Istar, prng)

	if s <=T and nTotal < nExpected:
		# attribute (weighted random sample, since sum(mu)==Istar)
		n0 = int(prng.choice(np.arange(dim), 1, p=(mu_endo / Istar)))
		history.append((n0, s, 0))
		nTotal += 1

	# value of \lambda(t_k) where k is most recent event
	# starts with just the base rate
	lastrates = mu_endo.copy()

	decIstar = False
	while nTotal < nExpected:
		if len(history) == 0:
			return history, 0.
		uj, tj = int (history[-1][0]), history[-1][1]

		if decIstar:
			# if last event was rejected, decrease Istar
			Istar = np.sum(rates)
			decIstar = False
		else:
			# otherwise, we just had an event, so recalc Istar (inclusive of last event)
			Istar = np.sum(lastrates) + alpha[uj,:].sum()

		s += drawExpRV (1. / Istar, prng)
		if s > T:
			break

		# calc rates at time s (use trick to take advantage of rates at last event)
		rates = mu_endo + kernel (s, tj, omega) * (alpha[uj, :] + lastrates - mu_endo)

		# attribution/rejection test
		# handle attribution and thinning in one step as weighted random sample
		diff = Istar - np.sum(rates)
		n0 = int (prng.choice(np.arange(dim+1), 1, p=(np.append(rates, diff) / Istar)))

		if n0 < dim:
			history.append((n0, s, 0))
			# update lastrates
			lastrates = rates.copy()
			nTotal += 1
		else:
			decIstar = True

	T = history[-1][1]
	return history, T


def simulate_mark(timestamps, timestamp_dims, beta, dims, v, num_sentiments=5):
	sentiments = np.linspace(-1, 1, num=num_sentiments)
	size = timestamps.size
	marks = np.empty(size)
	marks[0] = np.random.choice(sentiments)
	timestamps_beta = np.empty((dims, timestamps.size))
	intensity = np.empty(size)
	for i in range(dims):
		timestamps_beta[i] = beta[:, i][timestamp_dims]
	for i in range(1, size):
		dim = timestamp_dims[i]
		time_diff = timestamps[i] - timestamps[:i]
		intensity[i] = np.sum(timestamps_beta[dim, :i] * marks[:i] * np.exp(-v * time_diff))
		p = np.exp(sentiments * intensity[i]) / np.sum(np.exp(sentiments * intensity[i]))
		marks[i] = np.random.choice(sentiments, p=p)
	return marks
