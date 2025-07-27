# -----------------------------
# SIR Model with BayesFlow Inference
# -----------------------------
# This file implements a Susceptible-Infected-Recovered (SIR) model with Bayesian inference
# using the BayesFlow framework. It includes model definitions, training procedures,
# and MCMC refinement for parameter estimation.
# -----------------------------
# Imports and Dependencies
# -----------------------------
import os
import gc
import scipy
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import csv
import scipy
from scipy import integrate
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from bayesflow.helper_networks import EquivariantModule, InvariantModule
import seaborn as sns
import matplotlib.pyplot as plt
from bayesflow.default_settings import OPTIMIZER_DEFAULTS
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.keras.callbacks import Callback
from functools import partial
import time
from bayesflow.networks import InvertibleNetwork, SequenceNetwork
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, LayerNormalization
from tensorflow.keras.models import Sequential
from bayesflow import default_settings as defaults
from bayesflow.helper_networks import EquivariantModule, InvariantModule, MultiConv1D
import pickle
from scipy.stats import beta



# -----------------------------
# Model Parameters
# -----------------------------
N = 12800  # Population size
T_steps = 100  # Number of time steps
n = T_steps  # Alias for time steps

d = 2  # Dimension of parameter theta [lambda, mu]
d_x = 1  # Dimension of observation variable x
p = 5  # Dimension of summary statistics
Q = 1  # Number of draws for penalty term calculation
batch_size = 256  # Batch size for training

# Bayesflow hyperparameters
default_lr = 0.002 # 0.0005
epochs = 700
Np = 500 

# MCMC Parameters Setup
Ns = 5  # Number of draws for empirical likelihood estimator
N_proposal = 500  # Number of proposals for MCMC (run N_proposal chains)
burn_in = 199  # 199  # Number of burn-in steps
n_samples = 1
thin = 10
proposed_std = 0.05
quantile_level = 0.0025
epsilon_upper_bound = 0.02 # 0.02

# color setting and upper labels
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels = ["\\theta_1", "\\theta_2"]

# -----------------------------
# File Paths and Directories
# -----------------------------
current_dir = os.getcwd()

# Create output directories if they don't exist
fig_folder = "bf_fig"
os.makedirs(fig_folder, exist_ok=True)

gif_folder = "bf_gif"
os.makedirs(gif_folder, exist_ok=True)

bf_ps_folder = "bf_ps"
os.makedirs(bf_ps_folder, exist_ok=True)

debug_txt_path = os.path.join(current_dir, "bf_debug.txt")

# -----------------------------
# Random Number Generation and True Parameters
# -----------------------------
rng = np.random
true_ps = np.array([0.4, 0.15])
true_ps_tf = tf.constant([0.4, 0.15], dtype=tf.float32)

dtype = np.float32
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)  # Shape: (n * d_x)


# SIR Model Components


# Prior distribution function to generate parameters
def prior(batch_size):
    # Infection rate
    lambda_ = np.random.uniform(0, 1, size=batch_size)
    # Recovery rate
    mu_ = np.random.uniform(0, 1, size=batch_size)

    theta = np.stack([lambda_, mu_], axis=1)
    return theta


# Function to generate a single parameter sample from the prior
def _Prior():
    theta = prior(1)
    theta = np.reshape(theta, (d))
    return theta


# Function to generate observations based on parameters and time steps
def generate_observation(theta, T_steps):
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
        # Ensure observations are within [0, 1]

    return np.array(I_new_obs_list).T


# Simulator function to generate simulated data
def _simulator(theta):
    """
    :param theta: Shape must be (batch_size, 2).
                  batch_size can be any positive integer (including 2).
                  Each sample corresponds to a parameter combination [lambda_, mu_].
    :return: Shape is (batch_size, 100, 1).
    """
    # Ensure input is a 2D array (batch_size, 2)
    theta = np.atleast_2d(theta)

    # Generate batch observations
    X = generate_observation(theta, T_steps)  # Returns shape (batch_size, 100)

    # Reshape to (batch_size, 100, 1)
    return tf.reshape(X, [theta.shape[0], T_steps, d_x])


# Bayesflow Components


# -----------------------------
# Gated Recurrent Unit (GRU) summary network
# -----------------------------
class GRUSummary(keras.layers.Layer):
    """GRU-based time series summarization layer.

    Compresses time series data into fixed-length summary statistics using a GRU network.

    Args:
        gru_units (int): Number of units in the GRU layer
        summary_dim (int): Dimension of the output summary statistics
        dropout_rate (float): Dropout rate for regularization
    """

    def __init__(
        self,
        gru_units=64,
        summary_dim=8,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gru_units = gru_units
        self.summary_dim = summary_dim
        self.dropout_rate = dropout_rate

        self.gru = keras.layers.GRU(
            gru_units,
            dropout=dropout_rate,
        )
        self.norm = LayerNormalization()
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series, **kwargs):
        """Process time series data to generate summary statistics.

        Args:
            time_series (tf.Tensor): Input tensor of shape (batch_size, T, 1)

        Returns:
            tf.Tensor: Summary statistics of shape (batch_size, summary_dim)
        """
        # adapt to bayesflow trainer
        if len(time_series.shape) == 4:
            time_series = tf.squeeze(time_series, axis=1)

        # Process time series through GRU
        gru_output = self.gru(time_series, training=kwargs.get("stage") == "training")
        gru_output = self.norm(gru_output)
        # Generate final summary statistics
        summary = self.summary_stats(gru_output)
        return summary


## run experiment
def run_experiments(it):

    gru_units = 64
    dropout_rate = 0.1

    # Generate training set
    prior = bf.simulation.Prior(prior_fun=_Prior)
    simulator = bf.simulation.Simulator(simulator_fun=_simulator)
    bayesflow = bf.simulation.GenerativeModel(prior=prior, simulator=simulator)

    # Debug section
    summary_net = GRUSummary(
        gru_units=gru_units, summary_dim=p, dropout_rate=dropout_rate
    )
    inference_net = bf.networks.InvertibleNetwork(num_params=d, num_coupling_layers=5)
    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    trainer = bf.trainers.Trainer(
        amortizer=amortizer, generative_model=bayesflow, default_lr=default_lr
    )

    # View network structure
    # amortizer.summary()
    # amortizer.summary_net.summary()

    # Load pre-generated offline data
    file_path = os.path.join(current_dir, "data", f"x_train_bf.pkl")
    with open(file_path, "rb") as pickle_file:
        offline_data = pickle.load(pickle_file)

    print("offline_data sim data shape", offline_data["sim_data"].shape)

    # Start training
    start_train_time = time.time()
    trainer.train_offline(
        offline_data, epochs=epochs, batch_size=batch_size, validation_sims=batch_size
    )
    end_train_time = time.time()
    elapsed_train_time = end_train_time - start_train_time
    elapsed_train_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_train_time))

    # Clean up offline data
    del offline_data
    gc.collect()

    # Prepare target data
    x_target = obs_xs.reshape(1, n, d_x)

    # Sample from the posterior distribution
    bf_ps = amortizer.sample({"summary_conditions": x_target}, n_samples=Np)

    # Save the sampling results
    file_path = os.path.join(bf_ps_folder, f"bf_ps_{it}.npy")
    np.save(file_path, bf_ps)

    # Calculate bias
    bf_ps_mean = tf.reduce_mean(bf_ps, axis=0)
    bias = tf.norm(bf_ps_mean - true_ps_tf, ord="euclidean", axis=None)
    bias_vec = tf.abs(bf_ps_mean - true_ps_tf)

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

    # Set plot style
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    true_ps = [0.4, 0.15]

    # Set x-axis limits for each subplot
    x_limits = {
        0: (0, 0.6),
        1: (0, 0.3),
    }

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            bf_ps[:, j],
            ax=axs[j],
            fill=False,
            label="BF",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # Add vertical lines to indicate the true parameter values
    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # Calculate and plot credible intervals
    low, high = credible_interval(bf_ps)
    for i in range(d):
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color=est_color, alpha=0.3
        )  # Fill the credible interval area
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    # Save the figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    graph_file = os.path.join(fig_folder, f"sir_bf_experiment_{it}.png")
    plt.savefig(graph_file)
    plt.close()

    # -----------------------------
    # MCMC Refinement Overview
    # -----------------------------
    # Refinement using Monte Carlo ABC with weight being calculated as a kernel regression estimator or direct sample estimation
    # This section implements MCMC to refine the parameter estimation results.

    TX_target_ = summary_net(tf.convert_to_tensor(x_target, dtype=tf.float32))

    # -----------------------------
    # Calculate Bandwidth for Likelihood Estimator
    # -----------------------------
    N0 = 5000
    xx = tf.convert_to_tensor(x_target)
    xx = tf.tile(xx, [N0, 1, 1])
    Theta0 = amortizer.sample({"summary_conditions": x_target}, n_samples=N0)

    xn_0 = generate_observation(Theta0, T_steps)
    xn_0 = np.array(xn_0)
    xn_0 = xn_0.reshape(N0, n, d_x)
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)
    TT = summary_net(xn_0)
    Diff = tf.reduce_sum((summary_net(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, dtype=tf.float32)
    quan1 = np.quantile(Diff.numpy(), quantile_level)  
    quan1 = min(quan1, epsilon_upper_bound)  # Ensure quan1 does not exceed the upper bound
    quan1 = tf.constant(quan1, dtype=tf.float32)


    # -----------------------------
    # Create Folders for Saving Figures
    # -----------------------------

    # create a new folder under bf_gif_folder named "bf_gif_{it}"
    temp_gif_folder = os.path.join(gif_folder, f"bf_gif_{it}")
    # make the folder if it does not exist
    os.makedirs(temp_gif_folder, exist_ok=True)

    for i in range(d):
        theta_i_gif_folder = os.path.join(temp_gif_folder, f"theta_{i+1}")
        os.makedirs(theta_i_gif_folder, exist_ok=True)

    # -----------------------------
    # Plotting Function Definition
    # -----------------------------
    # Plot estimated proposals every 10 steps
    def plot(Theta_seq, bf_ps, true_ps, temp_gif_folder, steps, truncate_window=1):

        # Truncate the sequence to the last few proposals
        Theta_seq1 = tf.concat(Theta_seq[steps - truncate_window : (steps)], axis=0)
        Theta_est = Theta_seq1

        # 绘图
        sns.set_style("whitegrid")

        x_limits = {
            0: (0, 0.6),
            1: (0, 0.3),
        }

        for j in range(d):

            temp_j_gif_folder = os.path.join(temp_gif_folder, f"theta_{j + 1}")
            os.makedirs(temp_j_gif_folder, exist_ok=True)

            # Create a new figure
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(6, 4))

            if j in x_limits:
                ax.set_xlim(x_limits[j])
                ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

            # plot the KDE for the current parameter
            sns.kdeplot(
                bf_ps[:, j],
                ax=ax,
                fill=False,
                label="BF",
                color="blue",
                linewidth=1,
                linestyle="-",
            )
            sns.kdeplot(
                Theta_est[:, j],
                ax=ax,
                fill=False,
                label="BF+ABC-MCMC",
                color="green",
                linewidth=1,
                linestyle="-",
            )
            # Add a vertical line to indicate the true parameter value
            ax.axvline(true_ps[j], color="r", linestyle="--", linewidth=1)
            low, high = credible_interval(bf_ps)
            low_refined, high_refined = credible_interval(Theta_est)
            ax.axvline(low[j], color="b", linestyle="--", linewidth=1)
            ax.axvline(high[j], color="b", linestyle="--", linewidth=1)
            ax.axvline(low_refined[j], color="g", linestyle="--", linewidth=1)
            ax.axvline(high_refined[j], color="g", linestyle="--", linewidth=1)
            # Set title and labels
            ax.set_title(f"theta{j + 1} distributionat step {steps}")
            ax.set_xlabel(f"theta{j + 1}")
            ax.set_ylabel("Density")
            ax.legend()
            # Save the figure
            graph_file = os.path.join(
                temp_j_gif_folder, f"sir_bf_theta_{j+1}_steps_{steps}.png"
            )
            plt.savefig(graph_file)

    # -----------------------------
    # Function to Generate Initial MCMC Proposals
    # -----------------------------
    def generate_initial_proposal_mcmc(N_proposal):
        xx_proposal = tf.convert_to_tensor(x_target)
        xx_proposal = tf.tile(xx_proposal, [N_proposal, 1, 1])
        Theta_proposal = amortizer.sample(
            {"summary_conditions": x_target}, n_samples=N_proposal
        )
        Theta_proposal = tf.convert_to_tensor(Theta_proposal, dtype=tf.float32)
        # Truncate Theta_proposal to ensure values are within [0, 1]
        Theta_proposal = tf.clip_by_value(Theta_proposal, 0, 1)
        return Theta_proposal

    # -----------------------------
    # Function to Calculate Prior Density
    # -----------------------------
    def prior(theta):
        """Calculate the prior density of the parameters"""
        mask_ = tf.logical_and(theta >= 0.0, theta <= 1.0)
        prior_ = tf.cast(tf.reduce_prod(tf.cast(mask_, "float32"), axis=-1), "float32")
        return prior_

    def simulate_summary_data(theta, nsims):
        """Generate simulated data from the model and calculate summary statistics"""

        sim_X = np.zeros((theta.shape[0], nsims, n, d_x))

        theta_expand = tf.tile(tf.expand_dims(theta, axis=1), [1, nsims, 1])
        for i_sim in range(theta.shape[0]):
            sim_x_ = generate_observation(theta_expand[i_sim].numpy(), T_steps)
            sim_X[i_sim] = tf.expand_dims(sim_x_, axis=2)

        TX_ = np.zeros((theta.shape[0], nsims, p))
        for j_sim in range(nsims):
            TX_[:, j_sim, :] = summary_net(sim_X[:, j_sim, :, :]).numpy()
        TX_ = tf.convert_to_tensor(TX_, dtype=tf.float32)

        return TX_
    
    def distance(TX_sim, TX_target):
        """Calculate the distance between simulated data and target data"""
        return tf.reduce_sum((TX_sim - TX_target) ** 2, axis=-1)

    def approximate_likelihood(theta, nsims, epsilon):
        """Approximate the likelihood of the parameters"""
        TX_sim = simulate_summary_data(theta, nsims)
        dist = distance(TX_sim, TX_target_)
        kde_ = tf.reduce_mean(tf.exp(-dist / (2 * epsilon**2)), axis=-1)
        return kde_

    def mcmc_refinement(
        N_proposal=1,
        n_samples=5000,
        burn_in=2000,
        thin=10,
        nsims=50,
        epsilon=0.1,
        proposed_std=0.3,
    ):
        """Run ABC-MCMC sampling.

        Args:
            N_proposal (int, optional): Number of MCMC chains to run simultaneously. Defaults to 1.
            n_samples (int, optional): Total number of samples to generate. Defaults to 5000.
            burn_in (int, optional): Length of the burn-in period. Defaults to 2000.
            thin (int, optional): Sampling interval to reduce autocorrelation. Defaults to 10.
            nsims (int, optional): Number of simulations for likelihood approximation. Defaults to 50.
            epsilon (float, optional): Bandwidth parameter for likelihood approximation. Defaults to 0.1.
        """

        samples = []
        accepted = 0

        # Initialize MCMC proposals
        current_theta = generate_initial_proposal_mcmc(N_proposal)
        current_theta = tf.clip_by_value(
            current_theta, clip_value_min=0.0, clip_value_max=1.0
        )
        current_prior = prior(current_theta)
        current_likelihood = approximate_likelihood(current_theta, nsims, epsilon)
        current_ratio = current_prior * current_likelihood

        for i_mcmc in range(n_samples + burn_in):
            proposed_theta = current_theta + tf.random.normal(
                shape=current_theta.shape, mean=0, stddev=proposed_std
            )
            proposed_prior = prior(proposed_theta)
            proposed_likelihood = approximate_likelihood(proposed_theta, nsims, epsilon)
            proposed_ratio = proposed_prior * proposed_likelihood

            acceptance_prob = tf.minimum(1.0, proposed_ratio / current_ratio)
            u = tf.random.uniform(shape=(N_proposal,), minval=0.0, maxval=1.0)

            accept_mask = u < acceptance_prob
            accept_mask_2d = tf.expand_dims(accept_mask, axis=1)
            if i_mcmc >= burn_in:
                accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))

            current_theta = tf.where(accept_mask_2d, proposed_theta, current_theta)
            current_ratio = tf.where(accept_mask, proposed_ratio, current_ratio)

            if i_mcmc >= burn_in and (i_mcmc - burn_in) % thin == 0:
                samples.append(current_theta)

        acceptance_rate = accepted / (n_samples * N_proposal)
        samples = tf.concat(samples, axis=0)

        return samples, acceptance_rate

    # -----------------------------
    # Run MCMC Refinement Multiple Times
    # -----------------------------

    start_refined_time = time.time()
    Theta_mcmc, accp_rate = mcmc_refinement(
        N_proposal=N_proposal,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        nsims=Ns,
        epsilon=quan1,
        proposed_std=proposed_std,
    )
    refined_time_end = time.time()
    elapsed_refined_time = refined_time_end - start_refined_time
    elapsed_refined_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_refined_time)
    )
    
    # Save MCMC results
    mcmc_file_path = os.path.join(bf_ps_folder, f"bf_mcmc_{it}.npy")
    np.save(mcmc_file_path, Theta_mcmc.numpy())

    # -----------------------------
    # Calculate Bias for MCMC Results
    # -----------------------------

    Theta_mcmc_mean = tf.reduce_mean(Theta_mcmc, axis=0)
    refined_bias = tf.norm(Theta_mcmc_mean - true_ps_tf, ord="euclidean", axis=None)
    refined_bias_vec = tf.abs(Theta_mcmc_mean - true_ps_tf)
    refined_bias_vec = tf.cast(refined_bias_vec, "float32")

    # -----------------------------
    # Plot Final Posterior Estimation
    # -----------------------------


    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    true_ps = [0.4, 0.15]

    x_limits = {
        0: (0, 0.6),
        1: (0, 0.3),
    }

    # Set x-axis limits for each subplot
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels,range(d)):
        sns.kdeplot(
            bf_ps[:, j],
            ax=axs[j],
            fill=False,
            label="BF",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        sns.kdeplot(
            Theta_mcmc[:, j],
            ax=axs[j],
            fill=False,
            label="BF+ABC-MCMC",
            color=refined_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # Add vertical lines to indicate the true parameter values
    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    low_Y0, high_Y0 = credible_interval(bf_ps)
    low_Y0_refined, high_Y0_refined = credible_interval(Theta_mcmc)
    ci_lengths = high_Y0 - low_Y0
    ci_lengths_refined = high_Y0_refined - low_Y0_refined

    for i in range(d):
        axs[i].axvline(low_Y0[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high_Y0[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(low_Y0_refined[i], color=refined_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high_Y0_refined[i], color=refined_color, linestyle="--", linewidth=1.5)


    # Save the figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    graph_file = os.path.join(fig_folder, f"sir_bf_refined_experiment_{it}.png")
    plt.savefig(graph_file)
    plt.close()

    return (
        bias,
        refined_bias,
        elapsed_train_time_str,
        elapsed_refined_time_str,
        low_Y0,
        high_Y0,
        low_Y0_refined,
        high_Y0_refined,
        bias_vec,
        refined_bias_vec,
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    )


csv_file = "bf_sir_result1.csv"
credible_interval_file = "bf_sir_credible_interval1.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "train_time",
            "refined_time",
            "bias",
            "refined_bias",
            "bias_1",
            "bias_2",
            "refined_bias_1",
            "refined_bias_2",
            "acceptance_rate",
        ]
    )

with open(credible_interval_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "ci_length_1",
            "ci_length_2",
            "ci_length_refined_1",
            "ci_length_refined_2",
            "low_Y0_1",
            "high_Y0_1",
            "low_Y0_2",
            "high_Y0_2",
            "low_Y0_refined_1",
            "high_Y0_refined_1",
            "low_Y0_refined_2",
            "high_Y0_refined_2",
        ]
    )


for it in range(10):

    (
        bias,
        refined_bias,
        elapsed_train_time_str,
        elapsed_refined_time_str,
        low_Y0,
        high_Y0,
        low_Y0_refined,
        high_Y0_refined,
        bias_vec,
        refined_bias_vec,
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    ) = run_experiments(it)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                elapsed_train_time_str,
                elapsed_refined_time_str,
                bias.numpy(),
                refined_bias.numpy(),
                bias_vec[0].numpy(),
                bias_vec[1].numpy(),
                refined_bias_vec[0].numpy(),
                refined_bias_vec[1].numpy(),
                accp_rate.numpy(),
            ]
        )

    with open(credible_interval_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                *ci_lengths,
                *ci_lengths_refined,
                low_Y0[0],
                high_Y0[0],
                low_Y0[1],
                high_Y0[1],
                low_Y0_refined[0],
                high_Y0_refined[0],
                low_Y0_refined[1],
                high_Y0_refined[1],
            ]
        )
