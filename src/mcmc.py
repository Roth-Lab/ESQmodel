import numpy as np
from scipy import stats
from numba import jit


def update_mu(iter, prior, proposal, trace, y_data, mu_prior_means):
    # Sample mu
    for k in range(trace.k):
        for j in range(trace.j):
            new_mu = __sample_mu(k, j, prior, proposal, trace, y_data, mu_prior_means)
            trace.mu_trace[0, k, j] = new_mu
            trace.mu_trace[iter, k, j] = new_mu


def __sample_mu(cluster, marker, prior, proposal, trace, y_data, mu_prior_means):
    mu_j = trace.mu_trace[0, :, marker]
    var_j = trace.tau_trace[0, marker]
    rho = trace.rho_trace[0, :, :]

    # Current
    curr_mu = trace.mu_trace[0, cluster, marker]
    curr_log_prior = prior.log_density(curr_mu, loc=mu_prior_means[cluster, marker])
    curr_log_likelihood = mu_tau_log_likelihood(mu_j, var_j, rho, trace.i, trace.k, y_data[:, marker])

    # Proposed
    proposed_mu = proposal.propose(curr_mu)
    proposed_log_prior = prior.log_density(proposed_mu, loc=mu_prior_means[cluster, marker])
    mu_j[cluster] = proposed_mu
    proposed_log_likelihood = mu_tau_log_likelihood(mu_j, var_j, rho, trace.i, trace.k, y_data[:, marker])
    mu_j[cluster] = curr_mu

    # Accept/Reject
    u = np.log(stats.uniform.rvs())
    log_ratio = proposed_log_prior + proposed_log_likelihood - curr_log_prior - curr_log_likelihood

    if u < log_ratio:
        return proposed_mu
    else:
        return curr_mu


@jit(nopython=True)
def mu_tau_log_likelihood(mu_j, var_j, rho, N, K, y_j):
    log_likelihood = 0
    for i in range(N):
        y_ij = y_j[i]
        mu_sum = 0
        for k in range(K):
            mu_sum += rho[i, k] * mu_j[k]
        log_likelihood += -np.log(np.sqrt(var_j)) - np.log(np.sqrt(2 * np.pi)) - 0.5 * (y_ij - mu_sum) ** 2 / var_j
    return log_likelihood


def update_tau(iter, prior, proposal, trace, y_data):
    for j in range(trace.j):
        new_tau = __sample_tau(j, prior, proposal, trace, y_data)
        trace.tau_trace[0, j] = new_tau
        trace.tau_trace[iter, j] = new_tau


def __sample_tau(marker, prior, proposal, trace, y_data):
    mu_j = trace.mu_trace[0, :, marker]
    rho = trace.rho_trace[0, :, :]

    # Current
    curr_var = trace.tau_trace[0, marker]
    curr_log_prior = prior.log_density(curr_var)
    curr_log_likelihood = mu_tau_log_likelihood(mu_j, curr_var, rho, trace.i, trace.k, y_data[:, marker])

    # Proposed
    proposed_var = proposal.propose(curr_var)
    proposed_log_prior = prior.log_density(proposed_var)
    proposed_log_likelihood = mu_tau_log_likelihood(mu_j, proposed_var, rho, trace.i, trace.k, y_data[:, marker])

    # Accept/Reject
    u = np.log(stats.uniform.rvs())
    log_ratio = proposed_log_prior + proposed_log_likelihood - curr_log_prior - curr_log_likelihood

    if u < log_ratio:
        return proposed_var
    else:
        return curr_var


def update_rho(iter, proposal, trace, y_data):
    # With cache version
    curr_log_likelihood_cache = rho_log_likelihood(trace.mu_trace[0, :, :], trace.tau_trace[0, :],
                                                  trace.rho_trace[0, :, :], trace.i, trace.j, trace.k, y_data)
    for i in range(trace.i):
        new_rho, new_log_likelihood = ___sample_rho(i, proposal, trace, y_data, curr_log_likelihood_cache)
        curr_log_likelihood_cache = new_log_likelihood
        trace.rho_trace[0, i, :] = new_rho
        trace.rho_trace[iter, i, :] = new_rho


def ___sample_rho(cell, proposal, trace, y_data, curr_log_likelihood):
    mu = trace.mu_trace[0, :, :]
    var = trace.tau_trace[0, :]
    rho = trace.rho_trace[0, :, :]

    # Delta
    curr_rho = rho[cell, :]
    proposed_rho = proposal.propose(curr_rho)
    proposed_log_likelihood = delta_rho_log_likelihood(mu, var, curr_rho, proposed_rho, trace.j, trace.k,
                                                       y_data[cell, :], curr_log_likelihood)

    # Accept/Reject
    u = np.log(stats.uniform.rvs())
    log_ratio = proposed_log_likelihood - curr_log_likelihood

    if u < log_ratio:
        return proposed_rho, proposed_log_likelihood
    else:
        return curr_rho, curr_log_likelihood


@jit(nopython=True)
def rho_log_likelihood(mu, var, rho, N, J, K, y):
    log_likelihood = 0
    for i in range(N):
        for j in range(J):
            y_ij = y[i, j]
            var_j = var[j]
            mu_sum = 0
            for k in range(K):
                mu_sum += rho[i, k] * mu[k, j]
            log_likelihood += -np.log(np.sqrt(var_j)) - np.log(np.sqrt(2 * np.pi)) - 0.5 * (y_ij - mu_sum) ** 2 / var_j
    return log_likelihood


@jit(nopython=True)
def delta_rho_log_likelihood(mu, var, curr, prop, J, K, y, log_likelihood):
    curr_log_likelihood = 0
    prop_log_likelihood = 0
    for j in range(J):
        y_ij = y[j]
        var_j = var[j]
        curr_sum = 0
        prop_sum = 0
        for k in range(K):
            curr_sum += curr[k] * mu[k, j]
            prop_sum += prop[k] * mu[k, j]
        curr_log_likelihood += -np.log(np.sqrt(var_j)) - np.log(np.sqrt(2 * np.pi)) - 0.5 * (y_ij - curr_sum) ** 2 / var_j
        prop_log_likelihood += -np.log(np.sqrt(var_j)) - np.log(np.sqrt(2 * np.pi)) - 0.5 * (y_ij - prop_sum) ** 2 / var_j
    delta_log_likelihood = prop_log_likelihood - curr_log_likelihood
    new_likelihood = log_likelihood + delta_log_likelihood
    return new_likelihood
