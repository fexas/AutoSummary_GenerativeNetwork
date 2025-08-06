# SIR Model with Neural Network and MCMC Refinement
# This script implements a stochastic SIR model with neural network parameter estimation
# followed by MCMC refinement for improved posterior sampling.

# -----------------------------
# Imports and Dependencies
# -----------------------------
import os
import gc
import numpy as np
import tensorflow as tf
import math
import csv
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from functools import partial
import time
from tensorflow.keras.layers import LayerNormalization
from scipy.stats import beta

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"成功配置GPU: {physical_devices[0]}")
    except RuntimeError as e:
        print("GPU内存配置失败:", e)

# -----------------------------
# Configuration Parameters
# -----------------------------
# Model parameters
N = 12800  # Total population
T_steps = 100  # Number of time steps
n = T_steps  # Alias for time steps

d = 2  # Dimension of parameter theta (infection rate, recovery rate)
d_x = 1  # Dimension of observation variable x
p = 5  # Dimension of summary statistics
Q = 1  # Number of draws for penalty term

# Neural Network parameters
M = 50  # Number of theta estimates for MMD calculation
n_samples_z = 20  # Number of unit vectors in S^{d-1} for slicing
Np = 500  # 20000  # Number of theta estimates to generate
batch_size = 256  # Batch size for neural network training
default_lr = 0.002 # 0.003 # 0.001  # 0.00025  # 0.0005
epochs = 700

# MCMC Parameters Setup
Ns = 5  # Number of draws for empirical likelihood estimator
N_proposal = 500  # Number of proposals for MCMC (run N_proposal chains)
burn_in = 199  # 199  # Number of burn-in steps
n_samples = 1
thin = 10
proposed_std = 0.05
quantile_level = 0.0025
epsilon_upper_bound = 0.02 

# color setting and upper labels
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels = ["\\theta_1", "\\theta_2"]


# -----------------------------
# File Path Configuration
# -----------------------------
current_dir = os.getcwd()

# Create output directories if they don't exist
fig_folder = "nn_fig"
os.makedirs(fig_folder, exist_ok=True)

gif_folder = "nn_gif"
os.makedirs(gif_folder, exist_ok=True)

ps_folder = "nn_ps"
os.makedirs(ps_folder, exist_ok=True)

debug_txt_path = "nn_debug.txt"  # Debug output file
quan1_record_csv = "quan1_record.csv"

# -----------------------------
# Data Loading and Preparation
# -----------------------------
rng = np.random

# True parameter values (infection rate, recovery rate)
true_ps = np.array([0.4, 0.15])
true_ps_tf = tf.constant([0.4, 0.15], dtype=tf.float32)

dtype = np.float32
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)  # Shape: (n, d_x)
x_target = obs_xs.reshape(1, n, d_x)  # Reshape for model input


# -----------------------------
# SIR Model Functions
# -----------------------------


def prior(batch_size):
    """Generate samples from the prior distribution for SIR parameters.

    Args:
        batch_size (int): Number of parameter samples to generate

    Returns:
        np.ndarray: Sampled parameters with shape (batch_size, d)
    """
    lambda_ = np.random.uniform(0, 1, size=batch_size)  # Infection rate
    mu_ = np.random.uniform(0, 1, size=batch_size)  # Recovery rate

    theta = np.stack([lambda_, mu_], axis=1)
    return theta


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
        I_new_obs_list.append(I_new_obs) # Ensure observations are within [0, 1]

    return np.array(I_new_obs_list).T


def generate_dataset(theta_candidate, T_steps):
    X = generate_observation(theta_candidate, T_steps)
    X = np.array(X)
    X = X.astype(np.float32)
    return X


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

    def call(self, time_series, training=False):
        """Process time series data to generate summary statistics.

        Args:
            time_series (tf.Tensor): Input tensor of shape (batch_size, T, 1)
            training (bool): Whether the layer is in training mode

        Returns:
            tf.Tensor: Summary statistics of shape (batch_size, summary_dim)
        """
        # Process time series through GRU
        gru_output = self.gru(time_series, training=training)
        gru_output = self.norm(gru_output)
        # Generate final summary statistics
        summary = self.summary_stats(gru_output)
        return summary

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "gru_units": self.gru_units,
                "summary_dim": self.summary_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# -----------------------------
# Neural Network Definition
# -----------------------------


class NN(keras.Model):
    """Neural network model for SIR parameter estimation using MMD loss."""

    def __init__(self, G, T, **kwargs):
        super(NN, self).__init__(**kwargs)
        self.G = G  # Generator network
        self.T = T  # Summary network
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def MMD(self, theta, Gtheta):
        """
        Computes the Maximum Mean Discrepancy (MMD) between two sets of samples theta and Gtheta.
        Args:
            theta: [batch_size, d]
            Gtheta: [batch_size, M, d]
        """

        bandwidth = tf.constant([1 / n, 4 / n, 9 / n, 16 / n, 25 / n], "float32")
        bandwidth = tf.reshape(bandwidth, (bandwidth.shape[0], 1, 1, 1))

        theta_ = tf.expand_dims(theta, 1)  # (N,1,d)

        gg = tf.einsum("ijk,ikl->ijl", Gtheta, tf.transpose(Gtheta, perm=[0, 2, 1]))
        gt = tf.einsum("ijk,ikl->ijl", Gtheta, tf.transpose(theta_, perm=[0, 2, 1]))

        rg = tf.reduce_sum(tf.square(Gtheta), axis=2, keepdims=True)  # (N,M,1)
        rt = tf.reduce_sum(tf.square(theta_), axis=2, keepdims=True)  # (N,1,1)

        SE_gg = rg - 2 * gg + tf.transpose(rg, perm=[0, 2, 1])  # (N,M,M)
        SE_gt = rg - 2 * gt + tf.transpose(rt, perm=[0, 2, 1])  # (N,M,1)

        K_gg = tf.exp(-0.5 * tf.expand_dims(SE_gg, axis=0) / bandwidth)
        K_gt = tf.exp(-0.5 * tf.expand_dims(SE_gt, axis=0) / bandwidth)

        # mmd = tf.reduce_mean(K_gg) - 2 * tf.reduce_mean(K_gt)
        mmd = tf.reduce_mean(K_gg) * M**2 / (M * (M - 1)) - 2 * tf.reduce_mean(K_gt)

        return mmd

    def SliceMMD(self, theta, Gtheta):
        """
        Args:
        theta: [batch_size,d]
        Gtheta: [batch_size,M,d]

        """

        bandwidth = tf.constant(
            [1 / (2 * n)], "float32"
        )  # tf.constant([1 / n, 1 / (4 * n), 1 / (25 * n)], "float32")
        constant = tf.sqrt(1 / bandwidth)

        unit_vectors = tf.random.normal(shape=(d, n_samples_z))
        unit_vectors_norm = tf.norm(unit_vectors, axis=0, keepdims=True)
        unit_vectors = unit_vectors / unit_vectors_norm  # (d,L)

        Gtheta_diff = tf.expand_dims(Gtheta, 2) - tf.expand_dims(Gtheta, 1)
        slice_Gtheta_diff = tf.matmul(Gtheta_diff, unit_vectors)
        loss_term1 = tf.reduce_mean(
            constant[:, None, None, None, None]
            * tf.exp(
                -0.5
                * tf.square(slice_Gtheta_diff)
                / bandwidth[:, None, None, None, None]
            )
        )

        Gtheta_minus_theta = Gtheta - tf.expand_dims(theta, 1)
        slice_Gtheta_minus_theta = tf.matmul(Gtheta_minus_theta, unit_vectors)
        loss_term2 = tf.reduce_mean(
            constant[:, None, None, None]
            * tf.exp(
                -0.5
                * tf.square(slice_Gtheta_minus_theta)
                / bandwidth[:, None, None, None]
            )
        )

        # slice_MMD_loss = loss_term1 - 2 * loss_term2
        slice_MMD_loss = loss_term1 * M**2 / (M * (M - 1)) - 2 * loss_term2

        return slice_MMD_loss

    def train_step(self, x_train):
        """Perform one training step with MMD / Slice MMD calculation.

        Args:
            x_train: Input data batch

        Returns:
            dict: Dictionary containing loss metric
        """
        # Reshape input data
        x_train_ = tf.reshape(x_train, (batch_size, d + n * d_x))
        Theta_ = x_train_[:, 0:d]
        X_ = x_train_[:, d : n * d_x + d]
        X_ = tf.reshape(X_, (batch_size, n, d_x))

        # Generate normal random noise
        Z = tf.random.normal(shape=[batch_size, M, d])

        with tf.GradientTape() as tape:
            TX_ = self.T(X_, training=True)
            TX_ = tf.expand_dims(TX_, axis=1)
            TX_ = tf.tile(TX_, [1, M, 1])
            Z_and_TX = tf.concat((Z, TX_), axis=-1)
            Z_and_TX = tf.reshape(Z_and_TX, (batch_size * M, d + p))
            G_theta = self.G(Z_and_TX)
            G_theta = tf.reshape(G_theta, (batch_size, M, d))
            loss =  self.SliceMMD(theta=Theta_, Gtheta=G_theta) # self.MMD(theta=Theta_, Gtheta=G_theta) 

        grads = tape.gradient(loss, self.trainable_weights)

        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)

        return {"loss": self.total_loss_tracker.result()}


def run_experiments(it):
    """Main function to run SIR parameter estimation experiments.

    Args:
        it (int): Experiment iteration number
    """

    # -----------------------------
    # Summary Network Definition
    # -----------------------------
    gru_units = 64
    summary_dim = p

    # -----------------------------
    # Summary Netowrk
    # -----------------------------

    T = GRUSummary(gru_units=gru_units, summary_dim=summary_dim, dropout_rate=0.1)

    # -----------------------------
    # Generator Network Definition
    # -----------------------------
    intermediate_dim_G = 128
    G_inputs = keras.Input(shape=([p + d]))  # Input: z_i & T(x_{1:n}^i)

    # Create dense network with L1 regularization
    x = layers.Dense(
        units=intermediate_dim_G,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(G_inputs)
    # x = layers.Dense(
    #     units=128,
    #     activation="relu",
    #     kernel_regularizer=regularizers.l1(0.01),
    #     kernel_initializer="he_normal",
    # )(x)
    x = layers.Dense(
        units=64,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
    x = layers.Dense(
        units=32,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
    G_outputs = layers.Dense(units=d)(x)

    G = keras.Model(G_inputs, G_outputs, name="G")
    G.summary()

    nn = NN(G=G, T=T)

    # -----------------------------
    # Training Utilities
    # -----------------------------
    class LossHistory(Callback):
        """Callback to track training loss history."""

        def __init__(self):
            super().__init__()
            self.epoch_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_losses.append(logs["loss"])

    loss_history = LossHistory()

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        default_lr, epochs * batch_size, name="lr_decay"
    )

    OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}
    nn_optimizer = tf.keras.optimizers.Adam(schedule, **OPTIMIZER_DEFAULTS)

    nn.compile(optimizer=nn_optimizer, run_eagerly=False)

    # load training data
    file_path = os.path.join(current_dir, "data", "x_train.npy")
    x_train = np.load(file_path)

    train_time_start = time.time()

    nn.fit(
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[loss_history],
    )

    train_time_end = time.time()
    elapsed_train_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(train_time_end - train_time_start)
    )

    # delete x_train
    del x_train
    gc.collect()

    # plot and save losses
    plt.figure()
    plt.plot(loss_history.epoch_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_plot_path = os.path.join(fig_folder, f"training_loss_lr_{default_lr}_{it}.png")
    plt.savefig(loss_plot_path)
    plt.close()

    ## generate predicted theta
    xx = tf.convert_to_tensor(x_target, dtype=tf.float32)
    xx = tf.tile(xx, [Np, 1, 1])
    x = nn.T(xx)
    Y = tf.random.normal(shape=[Np, d])
    YY = tf.concat([Y, x], 1)
    Y0 = nn.G(YY)

    # save theta
    ps_path = os.path.join(ps_folder, f"nn_ps_{it}.npy")
    np.save(ps_path, Y0.numpy())

    # caluate bias
    Y0_mean = tf.reduce_mean(Y0, axis=0)
    bias = tf.norm(Y0_mean - true_ps_tf, ord="euclidean", axis=None)
    bias_vec = tf.abs(Y0_mean - true_ps_tf)

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

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    true_ps = [0.4, 0.15]

    x_limits = {
        0: (0, 0.6),
        1: (0, 0.3),
    }
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            Y0[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # plot 95% credible interval
    low, high = credible_interval(Y0)
    for i in range(d):
        # low, high = credible_interval(Y0)
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color=est_color, alpha=0.3
        )
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"sir_nn_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    # -----------------------------
    # MCMC Refinement Overview
    # -----------------------------
    # Refinement using Monte Carlo ABC with weight being calculated as a kernel regression estimator or direct sample estimation
    # This section implements MCMC to refine the parameter estimation results.

    TX_target_ = nn.T(tf.convert_to_tensor(x_target, dtype=tf.float32))

    # -----------------------------
    # Calculate Bandwidth for Likelihood Estimator
    # -----------------------------
    N0 = 5000
    xx = tf.convert_to_tensor(x_target, dtype=tf.float32)  # Shape: (1, n, d_x)
    xx = tf.tile(xx, [N0, 1, 1])  # Shape: (N0, n, d_x)
    x = nn.T(xx)  # Shape: (N0, p)
    Y = tf.random.normal(shape=[N0, d])
    YY = tf.concat([Y, x], 1)
    Theta0 = nn.G(YY)
    Theta0 = tf.cast(Theta0, "float32")
    Theta0 = tf.clip_by_value(Theta0, 0, 1)  # Ensure Theta0 is within [0,1]

    xn_0 = generate_observation(Theta0.numpy(), T_steps)
    xn_0 = np.array(xn_0)
    xn_0 = xn_0.reshape(N0, n, d_x)
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)

    TT = nn.T(xn_0)
    Diff = tf.reduce_sum((nn.T(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, "float32")
    quan1 = np.quantile(Diff.numpy(), quantile_level)
    # record the quan1 value in a CSV file
    with open(quan1_record_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([quan1])
    quan1 = min(quan1, epsilon_upper_bound)
    quan1 = tf.constant(quan1, dtype=tf.float32)

    # -----------------------------
    # Create Folders for Saving Figures
    # -----------------------------
    # Create a new folder under nn_gif_folder named 'nn_gif_{it}'
    temp_gif_folder = os.path.join(gif_folder, f"nn_gif_{it}")
    os.makedirs(temp_gif_folder, exist_ok=True)

    for i in range(d):
        theta_i_gif_folder = os.path.join(temp_gif_folder, f"theta_{i+1}")
        os.makedirs(theta_i_gif_folder, exist_ok=True)

    # -----------------------------
    # Plotting Function Definition
    # -----------------------------
    # Plot estimated proposals every 10 steps
    def plot(Theta_seq, Y0, true_ps, temp_gif_folder, steps, truncate_window=1):
        # Truncate the sequence to the last few proposals
        Theta_seq1 = tf.concat(Theta_seq[steps - truncate_window : steps], axis=0)
        Theta_est = Theta_seq1

        # Define x-axis limits for each parameter
        x_limits = {
            0: (0, 0.8),
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

            # Plot the KDE for the SMMD estimation
            sns.kdeplot(
                Y0[:, j],
                ax=ax,
                fill=False,
                label="SMMD",
                color="blue",
                linewidth=1,
                linestyle="-",
            )
            # Plot the KDE for the SMMD+ABC-MCMC estimation
            sns.kdeplot(
                Theta_est[:, j],
                ax=ax,
                fill=False,
                label="SMMD+ABC-MCMC",
                color="green",
                linewidth=1,
                linestyle="-",
            )

            # Add a vertical line to indicate the true parameter value
            ax.axvline(true_ps[j], color="r", linestyle="--", linewidth=1)

            # Calculate and plot credible intervals
            low_Y0, high_Y0 = credible_interval(Y0)
            low_Y0_refined, high_Y0_refined = credible_interval(Theta_est)
            ax.axvline(low_Y0[j], color="b", linestyle="--", linewidth=1)
            ax.axvline(high_Y0[j], color="b", linestyle="--", linewidth=1)
            ax.axvline(low_Y0_refined[j], color="g", linestyle="--", linewidth=1)
            ax.axvline(high_Y0_refined[j], color="g", linestyle="--", linewidth=1)

            # Set title and labels
            ax.set_title(f"theta{j + 1} distribution at step {steps}")
            ax.set_xlabel(f"theta{j + 1}")
            ax.set_ylabel("Density")
            ax.legend()

            # Save the figure
            graph_file = os.path.join(
                temp_j_gif_folder, f"sir_nn_steps_{steps}_theta_{j + 1}.png"
            )
            plt.savefig(graph_file)
            plt.close(fig)

    # -----------------------------
    # Function to Generate Initial MCMC Proposals
    # -----------------------------
    def generate_initial_proposal_mcmc(N_proposal):
        xx_proposal = tf.convert_to_tensor(x_target)
        xx_proposal = tf.tile(xx_proposal, [N_proposal, 1, 1])
        x_proposal = nn.T(xx_proposal)
        Y = tf.random.normal(shape=[N_proposal, d])
        YY = tf.concat([Y, x_proposal], 1)
        Theta_proposal = nn.G(YY)
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
            TX_[:, j_sim, :] = nn.T(sim_X[:, j_sim, :, :]).numpy()
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
            # if i_mcmc >= burn_in:
            #     accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))
            # Alternative
            accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))

            current_theta = tf.where(accept_mask_2d, proposed_theta, current_theta)
            current_ratio = tf.where(accept_mask, proposed_ratio, current_ratio)

            if i_mcmc >= burn_in and (i_mcmc - burn_in) % thin == 0:
                samples.append(current_theta)

        # acceptance_rate = accepted / (n_samples * N_proposal)
        # Alternative
        acceptance_rate = accepted / ( (n_samples + burn_in) * N_proposal)
        samples = tf.concat(samples, axis=0)

        return samples, acceptance_rate

    # -----------------------------
    # Run MCMC Refinement Multiple Times
    # -----------------------------

    refined_time_start = time.time()
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
    elapsed_refined_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(refined_time_end - refined_time_start)
    )

    # save MCMC results
    ps_path = os.path.join(ps_folder, f"nn_mcmc_{it}.npy")
    np.save(ps_path, Theta_mcmc.numpy())

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
    true_ps = [0.4, 0.15]

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    true_ps = [0.4, 0.15]

    x_limits = {
        0: (0, 0.6),
        1: (0, 0.3),
    }
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    # Plot KDE for SMMD+ABC-MCMC estimation
    for upper_label, j in zip(upper_labels, range(d)):

        sns.kdeplot(
            Y0[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )

        sns.kdeplot(
            Theta_mcmc[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD+ABC-MCMC",
            color=refined_color,
            linewidth=1.5,
            linestyle="-",
        )

        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # Add vertical lines to indicate true parameter values
    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # Bug fix: Change Theta_est to Theta_mcmc
    low_Y0, high_Y0 = credible_interval(Y0)
    low_Y0_refined, high_Y0_refined = credible_interval(Theta_mcmc)
    ci_lengths = high_Y0 - low_Y0
    ci_lengths_refined = high_Y0_refined - low_Y0_refined

    for i in range(d):
        axs[i].axvline(low_Y0[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high_Y0[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(
            low_Y0_refined[i], color=refined_color, linestyle="--", linewidth=1.5
        )
        axs[i].axvline(
            high_Y0_refined[i], color=refined_color, linestyle="--", linewidth=1.5
        )

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"sir_nn_refined_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    return (
        elapsed_train_time_str,
        elapsed_refined_time_str,
        bias,
        refined_bias,
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


csv_file = "nn_sir_result1.csv"
credible_interval_file = "nn_sir_credible_interval1.csv"


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
            "mcmc_accp_rate",
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

for it in range(10):  # 5 iterations with different learning rates

    (
        elapsed_train_time_str,
        elapsed_refined_time_str,
        bias,
        refined_bias,
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
        csv_writer = csv.writer(f)
        csv_writer.writerow(
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
        csv_writer = csv.writer(f)
        csv_writer.writerow(
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
