import os
import numpy as np
from pathlib import Path
import pandas as pd
from src.Trace import Trace
from src.Distributions import Gaussian, Gamma, Dirichlet
from src.mcmc import update_mu, update_tau, update_rho
from src.utils import marker_quantile, rho_entropy_record


def run(exp_csv,
        prior_csv,
        out_dir,
        num_iters=10000,
        ):

    # Constants
    os.makedirs(out_dir, exist_ok=True)
    y_data = pd.read_csv(exp_csv).to_numpy()
    prior = pd.read_csv(prior_csv).to_numpy()
    i_cells, j_markers = y_data.shape
    k_clusters = len(prior)
    kappa = 0.1
    output_dir = Path(out_dir)

    # Trace
    trace = Trace(t_iter=num_iters,
                  i_cells=i_cells,
                  j_markers=j_markers,
                  k_clusters=k_clusters)

    # Priors
    quantiles = marker_quantile(y_data)
    rank_dict = dict({1: 0, 2: 1, 3: 2})
    mu_prior_means = np.zeros((k_clusters, j_markers))
    for k in range(k_clusters):
        for j in range(j_markers):
            idx = rank_dict[prior[k, j]]
            mu_prior_means[k, j] = quantiles[j, idx]

    priors = {"mu": Gaussian(mean=0, var=0.001*2),
              "tau": Gamma(shape=1, loc=0, scale=1)}

    # Proposals
    proposals = {"mu": Gaussian(mean=0, var=0.1**2),
                 "tau": Gaussian(mean=0, var=0.01**2),
                 "rho": Dirichlet(alpha=[kappa] * k_clusters)}

    # Gibbs sampling
    for t in range(trace.t):
        print(f"Current iteration: {t}")
        update_mu(t, priors["mu"], proposals["mu"], trace, y_data, mu_prior_means)
        update_tau(t, priors["tau"], proposals["tau"], trace, y_data)
        update_rho(t, proposals["rho"], trace, y_data)

    # Calculate entropy
    rho_pt_est = trace.point_estimate(variable='rho')
    pd.DataFrame(rho_pt_est).to_csv(output_dir / "rho_est.csv", index=False)

    entropy = rho_entropy_record(rho_pt_est, "complete")
    pd.DataFrame(entropy).to_csv(output_dir / "entropy.csv", index=False)

    return trace
