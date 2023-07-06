import numpy as np
from scipy import stats


def marker_quantile(y_data):
    count = y_data.shape[1]
    rank = np.zeros((count, 3))
    for i in range(count):
        marker = y_data[:, i]
        marker = marker[~np.isnan(marker)]
        rank[i, 0] = np.quantile(marker, 0.25)
        rank[i, 1] = np.quantile(marker, 0.5)
        rank[i, 2] = np.quantile(marker, 0.75)
    return rank


def rho_entropy_record(rho, method="component"):
    entropy_arr = np.zeros(len(rho))
    if method == "component":
        for r in range(len(rho)):
            entropy_arr[r] = stats.entropy(rho[r]) / rho.shape[0]
        return entropy_arr
    elif method == "complete":
        for r in range(len(rho)):
            entropy_arr[r] = stats.entropy(rho[r])
        return entropy_arr
    else:
        raise Exception("Default method is component, else complete")
