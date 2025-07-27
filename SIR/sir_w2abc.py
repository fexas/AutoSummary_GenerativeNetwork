# SIR Model Posterior Estimation using Wasserstein ABC
# This script implements posterior estimation for SIR model parameters
# using Wasserstein distance-based Approximate Bayesian Computation
# refer https://github.com/seyni-diop/Bayesian-Learning-Project-Wasserstein-ABC/blob/master/WBAC_project.ipynb

# -----------------------------
# Imports and Dependencies
# -----------------------------
import os
from matplotlib.lines import lineStyles
import numpy as np
import tensorflow as tf
from math import *
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import ot
import scipy
from scipy.stats import beta
import csv
import sys

# -----------------------------
# Configuration Parameters
# -----------------------------
# Model parameters
N = 12800  # Total population
T_steps = 100  # Number of time steps
d_theta_universal = 2  # Dimension of parameter theta (infection rate, recovery rate)
d_x_universal = 1  # Dimension of observed data x (new infections)
true_ps = np.array([0.4, 0.15])
true_ps_list = [0.4, 0.15]

# parameters for reject sampling 
reject_sampling_n_samples = 500000

# File paths
current_dir = os.getcwd()
fig_folder = "w2abc_fig"
os.makedirs(fig_folder, exist_ok=True)
ps_folder = "w2abc_ps"
os.makedirs(ps_folder, exist_ok=True)
obs_data_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(obs_data_path)  # Shape: (n, d_x)
x_target = obs_xs.reshape(T_steps,)  # Target observations for SIR model

# tM matrix for wasserstein distance
u_times= np.arange(T_steps) + 1
v_times = np.arange(T_steps) + 1
u_times = u_times.astype(np.float32)
v_times = v_times.astype(np.float32)
tM = np.zeros((len(u_times), len(v_times)))
for i in range(len(u_times)):
    for j in range(len(v_times)):
        tM[i, j] = np.abs(u_times[i] - v_times[j])
        tM = np.power(tM, 1)

# color setting and upper labels
truth_color = "#FF6B6B"
est_color = "#4D96FF"
upper_labels = ["\\theta_1", "\\theta_2"]

# -----------------------------
# SIR Model Definition
# -----------------------------


class sir_prior:
    def __init__(self):
        pass

    def random(self, batch_size):
        """Generate samples from the prior distribution for SIR parameters."""
        lambda_ = np.random.uniform(0, 1, size=batch_size)  # Infection rate
        mu_ = np.random.uniform(0, 1, size=batch_size)  # Recovery rate
        return np.stack([lambda_, mu_], axis=1)

    def pdf(self, theta):
        """Compute the probability density function for SIR parameters."""
        if len(theta.shape) == 2:
            lambda_ = theta[:, 0]  # Infection rate
            mu_ = theta[:, 1]  # Recovery rate
            # 计算均匀分布的概率密度函数
            pdf_lambda = np.where((lambda_ >= 0) & (lambda_ <= 1), 1, 0)
            pdf_mu = np.where((mu_ >= 0) & (mu_ <= 1), 1, 0)
            pdf = pdf_lambda * pdf_mu
        elif len(theta.shape) == 1:
            lambda_ = theta[0]  # Infection rate
            mu_ = theta[1]  # Recovery rate
            # 计算均匀分布的概率密度函数
            pdf_lambda = np.where((lambda_ >= 0) & (lambda_ <= 1), 1, 0)
            pdf_mu = np.where((mu_ >= 0) & (mu_ <= 1), 1, 0)
            pdf = pdf_lambda * pdf_mu
        return pdf


def sir_sampler(theta, T_steps):
    """Generate observations from the SIR model with noise.

    Args:
        theta (np.ndarray): SIR parameters with shape (batch_size, d)
        T_steps (int): Number of time steps to simulate

    Returns:
        np.ndarray: Observed new infections with shape (batch_size, T_steps)
    """
    # Ensure input is 2D array
    theta = np.atleast_2d(theta)
    lambda_ = theta[:, 0]  # Infection rate
    mu_ = theta[:, 1]  # Recovery rate
    batch_size = theta.shape[0]

    # Batch initialization
    I = beta.rvs(1, 100, size=batch_size)  # Initial infected proportion
    S = 1 - I  # Initial susceptible proportion
    R = np.zeros_like(I)  # Initial recovered proportion
    sigma = 0.05  # Observation noise standard deviation

    I_new_obs_list = []

    for t in range(T_steps):
        I_new = lambda_ * S * I  # New infections

        # overflow protection
        I_new = np.where(I_new < S, I_new, S)

        S = S - I_new  # Update susceptible
        I = I + I_new - mu_ * I  # Update infected
        R = R + mu_ * I  # Update recovered

        # Add observation noise
        white_noise = np.random.normal(0, sigma, size=batch_size)
        I_new_obs = (1 + white_noise) * I_new
        I_new_obs = np.clip(I_new_obs, 0.0, 1.0)

        I_new_obs_list.append(I_new_obs)

    sir_observation = np.array(I_new_obs_list).T

    return sir_observation


# -----------------------------
# Wasserstein Distance Calculation
# -----------------------------
def t_fast_wasserstein(p, penalty_lambda):
    """
    Wasserstein distance for time series
    Inputs
    -----------
    p: order of the wasserstein distance

    Outputs
    -----------
    distance: function which return p-wasserstein distance between two sets
    """
    try:
        if p > 0:

            def distance(u_values, v_values):
                """
                Inputs
                ---------
                u_values: dataset, type array or list, shape = (n,.)
                v_values: dataset, type array or list, shape = (n,.)

                Outputs
                ---------
                dist: scalar, value of distance
                """

                a, b = ot.unif(len(u_values)), ot.unif(len(v_values))

                # covert all datatype to float 32
                u_values = u_values.astype(np.float32)
                v_values = v_values.astype(np.float32)

                if np.array(u_values)[0].shape != ():
                    M = np.power(ot.dist(u_values, v_values, metric="euclidean"), 1)
                else:
                    M = np.zeros((len(u_values), len(v_values)))
                    for i in range(len(u_values)):
                        for j in range(len(v_values)):
                            M[i, j] = np.abs(u_values[i] - v_values[j])
                    M = np.power(M, 1)

                M = M + penalty_lambda * tM
                M = np.power(M, p)

                dist = np.power(ot.emd2(a, b, M), 1 / p)
                return dist

    except:
        print("p must be positive")
    return distance


# -----------------------------
# Wasserstein ABC Algorithm
# -----------------------------



# reject sampling
def reject_sampling(y_true, y_sampler, prior, distance, n_samples, threshold):
    """
    Inputs
    -------------
    y_true : Observations (y observed) -- (T_steps, d_x_universal)
    y_sampler: function which takes two parameters theta at first and length
    prior: class
    distance: distance to use for ABC
    n_samples: number of particles
    ---------------
    Outputs:
    theta_list : np.array(), sampled particles
    list_epsilon: np.array(), distance between y_true and y_hat generated for all accepted samples
    --------------
    """
    theta_list = []
    accept_count = 0
    iteration_count = 0

    theta = prior.random(n_samples) # (n_samples, d_theta_universal)
    y_hat = y_sampler(theta, T_steps) # (n_samples, T_steps, d_x_universal)

    for i in range(n_samples):
        d = distance(y_hat[i], y_true)

        if d < threshold:
            theta_list.append(theta[i])
            accept_count += 1

    accp_rate = accept_count / n_samples

    return np.array(theta_list), accp_rate


# -----------------------------
# run SMC or reject sampling
# -----------------------------


def run_w2abc(it):
    """
    Inputs
    -------------
    it: number of iterations
    ---------------
    """

    wasserstein_p = 2  # order of wasserstein distance
    wasserstein_lambda = 2  # penalty coefficient for tfw2d


    prior = sir_prior()
    tfw2d = t_fast_wasserstein(wasserstein_p, wasserstein_lambda)

    # determine threshold
    theta_sample = prior.random(5000)  # Sample from prior to determine threshold
    observation_sample = sir_sampler(theta_sample, T_steps)  # Generate observations
    distances = np.array([
        tfw2d(obs,x_target) for obs in observation_sample
    ])  # Calculate distances
    reject_sampling_threshold = np.quantile(distances, 0.001)  # 1% quantile as threshold
    print("Threshold for Wasserstein distance:", reject_sampling_threshold)

    # -----------------------------
    # run reject sampling
    # -----------------------------
    
    print("Wasserstein ABC Reject Sampling starting...")    
    start_time = time.time()
    wasserstein_theta, accp_rate = reject_sampling(
        x_target,
        sir_sampler,
        prior,
        tfw2d,
        n_samples=reject_sampling_n_samples,
        threshold=reject_sampling_threshold,
    )
    end_time = time.time()
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))

    # save wasserstein_theta
    ps_path = os.path.join(ps_folder,f"w2abc_ps_{it}.npy")
    np.save(ps_path, wasserstein_theta)

    # calculate bias
    w2theta_mean = np.mean(wasserstein_theta, axis=0)
    bias = np.linalg.norm(w2theta_mean - true_ps, ord=2)
    bias_vec = np.abs(w2theta_mean - true_ps)

    # plot estimate posterior
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))


    x_limits = {
        0: (0, 0.6),
        1: (0, 0.3),
    }
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d_theta_universal)):
        sns.kdeplot(
            wasserstein_theta[:, j],
            ax=axs[j],
            label="W2ABC",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    for ax, true_p in zip(axs, true_ps_list):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # construct 95% credible interval for each parameter with empirical value Y0
    def credible_interval(Y0):
        """
        :param Y0: [Np, d]
        :return: [d, 2]
        """
        Np_temp = Y0.shape[0]
        Y0_temp = np.array(Y0)
        Y0_temp = np.sort(Y0_temp, axis=0)
        low = Y0_temp[int(0.025 * Np_temp), :]
        high = Y0_temp[int(0.975 * Np_temp), :]
        return low, high

    # plot 95% credible interval
    low, high = credible_interval(wasserstein_theta)
    ci_length = high - low
    for i in range(d_theta_universal):
        # low, high = credible_interval(Y0)
        axs[i].fill_betweenx(axs[i].get_ylim(), low[i], high[i], color=est_color, alpha=0.3)
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"sir_w2abc_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    return (
        elapsed_time_str,
        bias,
        bias_vec,
        low,
        high,
        accp_rate,  # Return acceptance rate for reject sampling
        ci_length,  # Return credible interval length
    )


csv_file = "w2_sir_result1.csv"
credible_interval_file = "w2_sir_credible_interval1.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "run_time",
            "bias",
            "bias_1",
            "bias_2",
            "acceptance_rate",  # Add acceptance rate column
        ]
    )

with open(credible_interval_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "ci_length_1",
            "ci_length_2",
            "low_Y0_1",
            "high_Y0_1",
            "low_Y0_2",
            "high_Y0_2",
        ]
    )

# -----------------------------
# Main Execution
# -----------------------------

for it in range(10):
    (
        elapsed_time_str,
        bias,
        bias_vec,
        low,
        high,
        accp_rate,  # Get acceptance rate from reject sampling
        ci_length,  # Get credible interval length
    ) = run_w2abc(it)

    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([it, elapsed_time_str, bias, bias_vec[0], bias_vec[1], accp_rate])
    with open(credible_interval_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([it, *ci_length, low[0], high[0], low[1], high[1]])

