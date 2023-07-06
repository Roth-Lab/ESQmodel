import numpy as np
from scipy import stats


class Gaussian:
    def __init__(self, mean, var):
        if var <= 0:
            raise ValueError("Variance <= 0")
        self.mean = mean
        self.var = var
        self.std = np.sqrt(self.var)

    def sample(self):
        return stats.norm.rvs(loc=self.mean, scale=self.std)

    def log_density(self, x, loc):
        if loc is None:
            loc = self.mean
        return stats.norm.logpdf(x, loc, scale=self.std)

    def propose(self, state):
        return stats.norm.rvs(loc=state, scale=self.std)


class Gamma:
    def __init__(self, shape, loc, scale):
        self.shape = shape
        self.loc = loc
        self.scale = scale

    def sample(self):
        return stats.gamma.rvs(a=self.shape, loc=self.loc, scale=self.scale)

    def log_density(self, x):
        if np.sum(x <= 0) > 0:
            return -float('inf')
        return stats.gamma.logpdf(x, a=self.shape, loc=self.loc, scale=self.scale)


class Dirichlet:
    def __init__(self, alpha):
        self.alpha = alpha
        self.n = len(alpha)

    def propose(self, state):
        proposal = np.zeros(self.n)
        for i in np.arange(self.n):
            proposal[i] = stats.gamma.rvs(a=0.1, loc=state[i], scale=1)
        proposal = proposal / np.sum(proposal)
        return proposal
