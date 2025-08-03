# Toy Model Posterior Estimation using Wasserstein ABC
# This script implements posterior estimation for Toy model parameters
# using Wasserstein distance-based Approximate Bayesian Computation
# refer https://github.com/seyni-diop/Bayesian-Learning-Project-Wasserstein-ABC/blob/master/WBAC_project.ipynb

# -----------------------------
# Imports and Dependencies
# -----------------------------
import os
import numpy as np
import tensorflow as tf
from math import *
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
# parameters for data
n = 25  # sample_size
d = 5  # dimension of parameter theta
d_x = 3  # dimenision of x
N_proposal = 5000000 # number of particles for proposal distribution

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4","\\theta_5"]

# File paths
current_dir = os.getcwd()
fig_folder = "w2abc_fig"
os.makedirs(fig_folder, exist_ok=True)
ps_folder = "w2abc_ps"
os.makedirs(ps_folder, exist_ok=True)
obs_data_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(obs_data_path)  # Shape: (n, d_x)
posterior_data_path = os.path.join(current_dir, "data", "ps.npy")
ps = np.load(posterior_data_path)
rng = np.random
file_path = os.path.join(current_dir, "data", "h_mmd.npy")
h_mmd = np.load(file_path)  # bandwidth of MMD
h_mmd = h_mmd**2

# -----------------------------
# Toy 50 Model Definition
# -----------------------------

class toy_prior:
    def __init__(self):
        pass

    def random(self, batch_size):
        """Generate samples from the prior distribution for toy model parameters."""
        theta = np.random.uniform(-3, 3, size=(batch_size, d))
        theta[:, 1] = theta[:, 0] ** 2 + np.random.randn(batch_size) * 0.1
        return theta


class SimulatorModel:
    """
    Base class for a simulator model.
    """

    def __init__(self):

        self.n_sims = 0

    def sim(self, ps):

        raise NotImplementedError("simulator model must be implemented as a subclass")


class Stats:
    """
    Identity summary stats.
    """

    def __init__(self):
        pass

    @staticmethod
    def calc(ps):
        return ps

    # prepare_cond_input

def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], "wrong sizes"

    return x, y, one_datapoint


class Model(SimulatorModel):
    """
    Simulator model.
    """

    def __init__(self):

        SimulatorModel.__init__(self)
        self.n_data = n

    def sim(self, ps, rng=np.random):
        """
        Simulate data at parameters ps.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return self.sim(ps[np.newaxis, :], rng=rng)[0]

        n_sims = ps.shape[0]

        m0, m1, s0, s1, r = self._unpack_params(ps)

        us = rng.randn(n_sims, self.n_data, 2)  # standard normal
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1

        self.n_sims += n_sims  # 9.10 -- 这么写n_sims原来应该是0

        return xs.reshape([n_sims, 2 * self.n_data])

    def sim_preserved_shape(self, ps, rng=np.random):
        """
        Simulate data at parameters ps.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return self.sim(ps[np.newaxis, :], rng=rng)[0]

        n_sims = ps.shape[0]

        m0, m1, s0, s1, r = self._unpack_params(ps)

        us = rng.randn(n_sims, self.n_data, 2)  # standard normal
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1

        self.n_sims += n_sims  # 9.10 -- 这么写n_sims原来应该是0

        return xs

    def eval(self, px, log=True):
        """
        Evaluate probability of data given parameters.
        """

        ps, xs, one_datapoint = prepare_cond_input(px, float)

        m0, m1, s0, s1, r = self._unpack_params(ps)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r**2)

        xs = xs.reshape([xs.shape[0], self.n_data, 2])
        us = np.empty_like(xs)

        us[:, :, 0] = (xs[:, :, 0] - m0) / s0
        us[:, :, 1] = (xs[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (
            s1 * np.sqrt(1.0 - r**2)
        )
        us = us.reshape([us.shape[0], 2 * self.n_data])

        L = np.sum(scipy.stats.norm.logpdf(us), axis=1) - self.n_data * logdet[:, 0]
        L = L[0] if one_datapoint else L

        return L if log else np.exp(L)

    @staticmethod
    def _unpack_params(ps):
        """
        Unpack parameters ps to m0, m1, s0, s1, r.
        """

        assert ps.shape[1] == 5, "wrong size"

        m0 = ps[:, [0]]
        m1 = ps[:, [1]]
        s0 = ps[:, [2]] ** 2
        s1 = ps[:, [3]] ** 2
        r = np.tanh(ps[:, [4]])

        return m0, m1, s0, s1, r
    
def stereo_proj(A):
    X_comp = A[..., 0]
    Y_comp = A[..., 1]

    # 计算新的分量
    new_X_comp = 2 * X_comp / (1 + X_comp**2 + Y_comp**2)
    new_Y_comp = 2 * Y_comp / (1 + X_comp**2 + Y_comp**2)
    Z_comp = (-1 + X_comp**2 + Y_comp**2) / (1 + X_comp**2 + Y_comp**2)

    result = np.stack([new_X_comp, new_Y_comp, Z_comp], axis=-1)

    return result
    
def toy_sampler(theta):
    observation = Stats().calc(Model().sim(theta, rng=rng))
    observation = observation.reshape(theta.shape[0], n, 2)  
    observation = stereo_proj(observation)  # Apply stereo projection
    return observation

# -----------------------------
# Wasserstein Distance Calculation
# -----------------------------
def fast_wasserstein(p):
    """
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
                if np.array(u_values)[0].shape != ():
                    M = np.power(ot.dist(u_values, v_values, metric="euclidean"), p)
                else:
                    M = np.zeros((len(u_values), len(v_values)))
                    for i in range(len(u_values)):
                        for j in range(len(v_values)):
                            M[i, j] = np.abs(u_values[i] - v_values[j])
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
    y_true : Observations (y observed)
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
    theta_proposal = prior.random(n_samples)
    y_hat_proposal = y_sampler(theta_proposal)
    
    for i in range(n_samples):
        d = distance(y_hat_proposal[i], y_true)

        if d < threshold:
            theta_list.append(theta_proposal[i])
            accept_count += 1
    
    accp_rate = accept_count / n_samples

    return np.array(theta_list), accp_rate

# -----------------------------
# Kernel and MMD Computation
# -----------------------------


# def compute_kernel(x, y, h_mmd):
#     x_size = tf.shape(x)[0]
#     y_size = tf.shape(y)[0]
#     dim = tf.shape(x)[1]

#     tiled_x = tf.tile(
#         tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
#     )
#     tiled_y = tf.tile(
#         tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
#     )
#     return tf.exp(-tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2) / (2 * h_mmd))


# def compute_mmd(x, y, h_mmd):
#     x_kernel = compute_kernel(x, x, h_mmd)
#     y_kernel = compute_kernel(y, y, h_mmd)
#     xy_kernel = compute_kernel(x, y, h_mmd)
#     return (
#         tf.reduce_mean(x_kernel)
#         + tf.reduce_mean(y_kernel)
#         - 2 * tf.reduce_mean(xy_kernel)
#     )

def compute_mmd(x, y, h_mmd):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    

    xx = tf.matmul(x, x, transpose_b=True)
    xy = tf.matmul(x, y, transpose_b=True)
    yy = tf.matmul(y, y, transpose_b=True)

    rx = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    ry = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)

    se_xx = rx - 2 * xx + tf.transpose(rx)
    se_xy = rx - 2 * xy + tf.transpose(ry)
    se_yy = ry - 2 * yy + tf.transpose(ry)

    kernel_xx = tf.exp(-se_xx / (2 * h_mmd))
    kernel_xy = tf.exp(-se_xy / (2 * h_mmd))
    kernel_yy = tf.exp(-se_yy / (2 * h_mmd))

    mmd = tf.reduce_mean(kernel_xx) + tf.reduce_mean(kernel_yy) - 2 * tf.reduce_mean(kernel_xy)

    return mmd


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

    x_target  = obs_xs.reshape(1,n,2)
    x_target = stereo_proj(x_target)  # Apply stereo projection
    x_target = x_target.reshape(n,d_x)

    prior = toy_prior()
    fw2d = fast_wasserstein(2)

    # determine threshold
    theta_sample = prior.random(5000)
    observation_sample = toy_sampler(theta_sample)
    distances = np.array([fw2d(obs, x_target) for obs in observation_sample])
    threshold = np.quantile(distances, 0.001)  # 0.1% quantile as threshold
    print("Threshold for Wasserstein distance:", threshold)

    time_start = time.time()
    wasserstein_theta, accp_rate = reject_sampling(
        y_true=x_target,
        y_sampler=toy_sampler,
        prior=prior,
        distance=fw2d,
        n_samples=N_proposal,
        threshold=threshold
    )
    time_end = time.time()
    time_taken = time_end - time_start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time_taken))

    w2_theta_path = os.path.join(ps_folder,f"w2_ps_{it}.npy")
    np.save(w2_theta_path,wasserstein_theta)

    mmd_w2 = compute_mmd(
        tf.convert_to_tensor(wasserstein_theta, dtype=tf.float32),
        tf.convert_to_tensor(ps, dtype=tf.float32),
        h_mmd,  
    )

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 5, figsize=(25, 6))
    true_ps = [1, 1, -1.0, -0.9, 0.6]

    # 定义每个theta_i对应的x轴范围
    x_limits = [
        [0.7, 1.3],  # theta_0
        [0.6, 1.4],  # theta_1
        [-1.5, 1.5],  # theta_2
        [-1.5, 1.5],  # theta_3
        [0, 1.2],  # theta_4
    ]

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            ps[:, j],
            ax=axs[j],
            fill=False,
            label="posterior",
            color=truth_color,
            linestyle="-.",
            linewidth=1.5,
        )
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

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    fig_path = os.path.join(fig_folder, f"w2abc_{it}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    return (
        elapsed_time_str,
        mmd_w2.numpy(),  # Bias of W2ABC
        accp_rate,  # Return acceptance rate for reject sampling
    )


csv_file = f"w2_{n}_result1.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "run_time",
            "mmd",
            "acceptance_rate",  # Add acceptance rate column
        ]
    )


# -----------------------------
# Main Execution
# -----------------------------

for it in range(10):
    (
        elapsed_time_str,
        mmd,  # Get MMD for W2ABC
        accp_rate,  # Get acceptance rate from reject sampling
    ) = run_w2abc(it)

    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([it, elapsed_time_str, mmd, accp_rate])
