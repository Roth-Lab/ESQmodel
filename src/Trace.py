import numpy as np
import matplotlib.pyplot as plt


class Trace:
    def __init__(self, t_iter, i_cells, j_markers, k_clusters, random_init=True):
        self.t = t_iter
        self.i = i_cells
        self.j = j_markers
        self.k = k_clusters
        self.mu_trace = np.zeros((self.t, self.k, self.j))
        self.tau_trace = np.ones((self.t, self.j))
        self.rho_trace = np.ones((self.t, self.i, self.k)) / self.k

        if random_init:
            self.mu_trace[0, :, :] = np.random.rand(self.k * self.j).reshape((self.k, self.j))
            self.tau_trace[0, :] = np.random.rand(self.j)
            rho_unnormalized = np.random.rand(self.i * self.k).reshape((self.i, self.k))
            self.rho_trace[0, :, :] = np.array([row / np.sum(row) for row in rho_unnormalized])

    def plot_trace(self, variable, output_dir):
        if variable == "mu":
            # plot t:mu[0, :], j lines for first cluster
            plt.figure(figsize=(3.8, 3.5), dpi=500)
            plt.plot(range(1, len(self.mu_trace)), self.mu_trace[1:, 0, :])
            plt.xlabel("Iterations", fontsize=8)
            plt.ylabel("Mu", fontsize=8)
            plt.title(f"iter:mu[0, :], show j lines for 0th cluster")
            plt.savefig(output_dir / f"run_mu_trace.png")
            plt.clf()

        elif variable == "tau":
            # plot t:tau, j lines
            plt.figure(figsize=(3.8, 3.5), dpi=500)
            plt.plot(range(1, len(self.tau_trace)), self.tau_trace[1:, :])
            plt.xlabel("Iterations", fontsize=8)
            plt.ylabel("Tau", fontsize=8)
            plt.title(f"iter:tau, show j lines")
            plt.savefig(output_dir / f"run_tau_trace.png")
            plt.clf()

        elif variable == "rho":
            # plot t:rho[0, :], k lines for first cell
            plt.figure(figsize=(3.8, 3.5), dpi=500)
            plt.plot(range(1, len(self.rho_trace)), self.rho_trace[1:, 0, :])
            plt.xlabel("Iterations", fontsize=8)
            plt.ylabel("Rho", fontsize=8)
            plt.title(f"iter:rho[0, :], show K lines for 0th cell")
            plt.savefig(output_dir / f"run_rho_trace.png")
            plt.clf()

        else:
            raise Exception("Not valid variable")

    def point_estimate(self, variable):
        # Average after removing half of burnin
        if variable == "mu":
            pt_est = np.zeros((self.k, self.j))
            for k in range(self.k):
                for j in range(self.j):
                    pt_est[k, j] = np.average(self.mu_trace[self.t//2:, k, j])
            return pt_est

        elif variable == "tau":
            pt_est = np.zeros(self.j)
            for j in range(self.j):
                pt_est[j] = np.average(self.tau_trace[self.t//2:, j])
            return pt_est

        elif variable == "rho":
            pt_est = np.zeros((self.i, self.k))
            for i in range(self.i):
                for k in range(self.k):
                    pt_est[i, k] = np.average(self.rho_trace[self.t//2:, i, k])
            for i in range(self.i):
                pt_est[i, :] = pt_est[i, :] / np.sum(pt_est[i, :])
            return pt_est

        else:
            raise Exception("Not valid variable")
