# LV Model Posterior Estimation using Wasserstein ABC
# This script implements posterior estimation for Lotka-Volterra model parameters
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
import ot
import csv
from scipy.integrate import odeint
import warnings

# -----------------------------
# Configuration Parameters
# -----------------------------
# parameters for data
N = 12800
n_points = 200  # n_points 貌似是取代了原来n的位置
n = n_points  # sample_size
d = 4  # dimension of parameter theta
d_x = 2  # dimenision of x
p = 9  # dimension of summary statistics
Q = 1  # number of draw from \exp{\frac{\Vert \theta_i - \theta_j \Vert^2}{w}} in first penalty
batch_size = 256

# params for lv model
T_count = 15
x0 = 10
y0 = 5
p_lower = -5.0
p_upper = 2.0
p_lower_list = np.array([p_lower] * d)
p_upper_list = np.array([p_upper] * d)
true_ps = np.log(np.array([1, 0.01, 0.5, 0.01]))
true_ps_tf = tf.convert_to_tensor(true_ps, dtype=tf.float32)
true_ps_ls = true_ps.tolist()

# tM matrix for wasserstein distance
u_times= np.arange(n_points) + 1
v_times = np.arange(n_points) + 1
u_times = u_times.astype(np.float32)
v_times = v_times.astype(np.float32)
tM = np.zeros((len(u_times), len(v_times)))
for i in range(len(u_times)):
    for j in range(len(v_times)):
        tM[i, j] = np.abs(u_times[i] - v_times[j])
        tM = np.power(tM, 1)

# params for w2abc
rng = np.random
N_proposal = 500000

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4"]

# File paths
current_dir = os.getcwd()
fig_folder = "w2abc_fig"
os.makedirs(fig_folder, exist_ok=True)
ps_folder = "w2abc_ps"
os.makedirs(ps_folder, exist_ok=True)
obs_data_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(obs_data_path)  # Shape: (n, d_x)


# -----------------------------
# lv Model Definition
# -----------------------------

class lv_prior:
    def __init__(self):
        pass

    def random(self, batch_size):
        """Generate samples from the prior distribution for Lotka-Volterra model parameters."""
        theta = np.random.uniform(p_lower_list, p_upper_list, size=(batch_size, 4))
        return theta

def lotka_volterra_forward(params, n_obs, T_span, x0, y0):

    def lotka_volterra_equations(state, t, alpha, beta, gamma, delta):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = -gamma * y + delta * x * y
        return [dxdt, dydt]

    def ecology_model(
        alpha, beta, gamma, delta, t_span=[0, 5], t_steps=100, initial_state=[1, 1]
    ):
        t = np.linspace(t_span[0], t_span[1], t_steps)
        state = odeint(
            lotka_volterra_equations, initial_state, t, args=(alpha, beta, gamma, delta)
        )
        x, y = state.T  # Transpose to get x and y arrays

        return (
            x,  # Prey time series
            y,  # Predator time series
            t,  # time
        )

    # parameter for the Lotka-Volterra model
    t_steps = n_obs
    t_span = [0, T_span]
    alpha, beta, gamma, delta = np.exp(params)
    initial_state = [x0, y0]
    noise_scale = 0.01

    x, y, t = ecology_model(
        alpha,
        beta,
        gamma,
        delta,
        t_span=t_span,
        t_steps=t_steps,
        initial_state=initial_state,
    )

    # add noise to the time series
    x += rng.normal(0, noise_scale, size=x.shape)
    y += rng.normal(0, noise_scale, size=y.shape)

    # concatenate the observed time series of x and y
    observed_X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    return observed_X


def lv_sampler(theta):
    theta = np.atleast_2d(theta)
    X = np.zeros((theta.shape[0], n_points, 2))

    for j in range(theta.shape[0]):
        X[j] = lotka_volterra_forward(theta[j], n_points, T_count, x0, y0)
    return X


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
# run SMC or reject sampling
# -----------------------------


def run_w2abc(it):
    """
    Inputs
    -------------
    it: number of iterations
    ---------------
    """

    x_target  = obs_xs.reshape(n,d_x)
    wasserstein_p = 2  # order of wasserstein distance
    wasserstein_lambda = 2  # penalty coefficient for tfw2d

    prior = lv_prior()
    tfw2d = t_fast_wasserstein(wasserstein_p, wasserstein_lambda)

    # determine threshold
    theta_sample = prior.random(5000)
    observation_sample = lv_sampler(theta_sample)
    distances = np.array([tfw2d(obs, x_target) for obs in observation_sample])
    threshold = np.quantile(distances, 0.001)  # 0.1% quantile as threshold
    print("Threshold for Wasserstein distance:", threshold)
    # if threshold is nan, using warning and return directly
    if np.isnan(threshold):
        warnings.warn("Threshold is NaN. Returning directly.")
        return

    time_start = time.time()
    wasserstein_theta, accp_rate = reject_sampling(
        y_true=x_target,
        y_sampler=lv_sampler,
        prior=prior,
        distance=tfw2d,
        n_samples=N_proposal,
        threshold=threshold
    )
    time_end = time.time()
    time_taken = time_end - time_start
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time_taken))

    # save wasserstein_theta
    ps_path = os.path.join(ps_folder,f"w2abc_ps_{it}.npy")
    np.save(ps_path, wasserstein_theta)

    # calculate bias
    w2theta_mean = np.mean(wasserstein_theta, axis=0)
    bias = np.linalg.norm(w2theta_mean - true_ps, ord=2)
    bias_vec = np.abs(w2theta_mean - true_ps)


    def credible_interval(Y0):
        """
        :param Y0: [Np, d]
        :return: [d, 2]
        """
        Np_temp = Y0.shape[0]
        # if Y0 is a tensor, convert it to numpy array
        if isinstance(Y0, tf.Tensor):
            Y0_temp = Y0.numpy()
        else:
            Y0_temp = Y0
        Y0_temp = np.sort(Y0_temp, axis=0)
        low = Y0_temp[int(0.025 * Np_temp), :]
        high = Y0_temp[int(0.975 * Np_temp), :]
        return low, high

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    x_limits = [
        [-0.4, 0.4],  # theta_0
        [-5.2, -4.0],  # theta_1
        [-0.85, -0.55],  # theta_2
        [-5.2, -4.0],  # theta_3
    ]
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            wasserstein_theta[:, j],
            ax=axs[j],
            fill=False,
            label="W2ABC",
            color=est_color,
            linestyle="-",
            linewidth=1.5,
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps_ls):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # 在每个子图上绘制 95% credible interval
    low, high = credible_interval(wasserstein_theta)
    credible_interval_length = high - low
    for i in range(d):
        # low, high = credible_interval(Y0)
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color=est_color, alpha=0.3
        )  # 填充带状区域
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    fig_path = os.path.join(fig_folder, f"w2abc_{it}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    return (
        elapsed_time_str,
        bias,  # Bias of W2ABC
        bias_vec,  # Bias vector of W2ABC
        accp_rate,  # Return acceptance rate for reject sampling
        low,  # Lower bound of credible interval
        high,  # Upper bound of credible interval
        credible_interval_length,  # Length of credible interval
    )


csv_file = "w2_lv_result1.csv"
credible_interval_file = "w2_lv_credible_interval1.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "run_time",
            "bias",
            "acceptance_rate",  # Add acceptance rate column
            "bias_theta1",
            "bias_theta2",
            "bias_theta3",
            "bias_theta4",
        ]
    )

with open(credible_interval_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "interval_length_theta1",
            "interval_length_theta2",
            "interval_length_theta3",
            "interval_length_theta4",
            "low_Y0_1",
            "high_Y0_1",
            "low_Y0_2",
            "high_Y0_2",
            "low_Y0_3",
            "high_Y0_3",
            "low_Y0_4",
            "high_Y0_4",
        ]
    )

# -----------------------------
# Main Execution
# -----------------------------

for it in range(10):
    (
        elapsed_time_str,
        bias, 
        bias_vec,  # Bias vector of W2ABC
        accp_rate,  # Get acceptance rate from reject sampling
        low,  # Lower bound of credible interval
        high,  # Upper bound of credible interval
        credible_interval_length,  # Length of credible interval
    ) = run_w2abc(it)

    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([it, elapsed_time_str, bias, accp_rate, *bias_vec])

    with open(credible_interval_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [it, *credible_interval_length, 
                low[0], high[0],
                low[1], high[1],
                low[2], high[2],
                low[3], high[3]]
            )
