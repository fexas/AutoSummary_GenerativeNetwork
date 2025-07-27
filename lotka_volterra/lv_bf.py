import os
import gc
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import csv
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle

N = 12800
n_points = 200  # n_points 貌似是取代了原来n的位置
n = n_points  # sample_size
d = 4  # dimension of parameter theta
d_x = 2  # dimenision of x
p = 9  # dimension of summary statistics
T = 15
x0 = 10
y0 = 5
p_lower = -5.0
p_upper = 2.0
T = 15
p_lower_list = np.array([p_lower] * d)
p_upper_list = np.array([p_upper] * d)

batch_size = 256
default_lr = 0.002 # 0.0005
epochs = 450
Np = 500

# MCMC Parameters Setup
Ns = 5  # 5
N_proposal = 500  # 3000
burn_in = 199
n_samples = 1
thin = 10
proposed_std = 0.05  # 1 / (2 * math.sqrt(n))
quantile_level = 0.0025
epsilon_upper_bound = 0.02

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4"]

# load data
current_dir = os.getcwd()

fig_folder = "bf_fig"
os.makedirs(fig_folder, exist_ok=True)

bf_ps_folder = "bf_ps"
os.makedirs(bf_ps_folder, exist_ok=True)
debug_txt_path = os.path.join(current_dir, "bf_debug.txt")

rng = np.random
true_ps = np.log(np.array([1, 0.01, 0.5, 0.01]))
true_ps_tf = tf.convert_to_tensor(true_ps, dtype=tf.float32)
true_ps_ls = true_ps.tolist()

dtype = np.float32
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)  # (n, d_x)


# lv model


def _Prior():
    theta = np.random.uniform(p_lower_list, p_upper_list)
    return np.exp(theta)

def lotka_volterra_forward(params,n_obs,T_span,x0,y0):

    def lotka_volterra_equations(state, t, alpha, beta, gamma, delta):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = - gamma * y + delta * x * y
        return [dxdt, dydt]

    def ecology_model(alpha, beta, gamma, delta, t_span=[0, 5], t_steps=100, initial_state=[1, 1]):
        t = np.linspace(t_span[0], t_span[1], t_steps)
        state = odeint(lotka_volterra_equations, initial_state, t, args=(alpha, beta, gamma, delta))
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


def simulate_lv_params(
    theta,
    n_points=n_points,
    x0=x0,
    y0=y0,
    T=T,
    to_tensor=True,
):
    """
    Simulates a batch of Ricker datasets given parameters.
    """

    theta = np.atleast_2d(theta)
    
    X = np.zeros((theta.shape[0], n_points, 2))

    for j in range(theta.shape[0]):
        X[j] = lotka_volterra_forward(theta[j], n_points, T, x0, y0)

    if to_tensor:
        return tf.convert_to_tensor(X, dtype=tf.float32)
    return X


def _simulator(theta, n_points=n_points, T=T, x0=x0, y0=y0):
    X = lotka_volterra_forward(theta, n_points, T, x0, y0)
    return X


## Bayesflow


def kronecker_product(mat1, mat2):
    """Computes the Kronecker product two matrices."""
    m1, n1 = mat1.get_shape().as_list()
    mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
    m2, n2 = mat2.get_shape().as_list()
    mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
    return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def compute_kernel1(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]

    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)


## run experiment
def run_experiments(it):

    # generate training set
    prior = bf.simulation.Prior(prior_fun=_Prior)

    simulator = bf.simulation.Simulator(simulator_fun=_simulator)

    bayesflow = bf.simulation.GenerativeModel(prior=prior, simulator=simulator)

    summary_net = bf.networks.SequenceNetwork(summary_dim=p)

    inference_net = bf.networks.InvertibleNetwork(num_params=d, num_coupling_layers=5)

    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

    trainer = bf.trainers.Trainer(
        amortizer=amortizer, generative_model=bayesflow, default_lr=default_lr
    )

    # 查看网络结构
    # amortizer.summary()
    # amortizer.summary_net.summary()

    # offline data 替换成事先生成好的数据

    file_path = os.path.join(current_dir, "data", f"x_train_bf.pkl")
    with open(file_path, "rb") as pickle_file:
        offline_data = pickle.load(pickle_file)

    start_train_time = time.time()
    trainer.train_offline(
        offline_data, epochs=epochs, batch_size=batch_size, validation_sims=batch_size
    )
    end_train_time = time.time()
    elapsed_train_time = end_train_time - start_train_time
    elapsed_train_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_train_time))

    # 清除offline data
    del offline_data
    gc.collect()

    x_target = obs_xs.reshape(1, n, d_x)

    bf_ps = amortizer.sample({"summary_conditions": x_target}, n_samples=Np)

    # save bf_ps
    bf_ps_path = os.path.join(bf_ps_folder, f"bf_lv_ps_{it}.npy")
    np.save(bf_ps_path, bf_ps)

    # 计算bias
    bf_ps_mean = tf.reduce_mean(bf_ps, axis=0)
    bias = tf.norm(bf_ps_mean - true_ps_tf, ord="euclidean", axis=None)

    def credible_interval(Y0):
        """
        :param Y0: [Np, d]
        :return: [d, 2]
        """
        Np_temp = Y0.shape[0]
        # if Y0 is tensor, convert it to numpy array
        if isinstance(Y0, tf.Tensor):
            Y0_temp = Y0.numpy()
        else:
            Y0_temp = Y0
        Y0_temp = np.sort(Y0_temp, axis=0)
        low = Y0_temp[int(0.025 * Np_temp), :]
        high = Y0_temp[int(0.975 * Np_temp), :]
        return low, high

    # 绘图
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
            bf_ps[:, j],
            ax=axs[j],
            label="BF",
            color=est_color,
            linestyle="-",
            linewidth=1.5,
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps_ls):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    low, high = credible_interval(bf_ps)
    for i in range(4):
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color=est_color, alpha=0.3
        )  # 填充带状区域
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    graph_file = os.path.join(fig_folder, f"bf_lv_{it}.png")
    plt.savefig(graph_file)
    plt.close()

    
    # -----------------------------
    # MCMC Refinement Overview
    # -----------------------------
    # Refinement using Monte Carlo ABC with weight being calculated as a kernel regression estimator or direct sample estimation
    # This section implements MCMC to refine the parameter estimation results.    
    
    TX_target_ = summary_net(x_target)  # (1, p)
    # -----------------------------
    # Calculate Bandwidth for Likelihood Estimator
    # -----------------------------
    N0 = 5000
    xx = tf.convert_to_tensor(x_target)
    xx = tf.tile(xx, [N0, 1, 1])
    Theta0 = amortizer.sample({"summary_conditions": x_target}, n_samples=N0)

    xn_0 = simulate_lv_params(Theta0, n_points=n_points, x0=x0, y0=y0, T=T, to_tensor=False)
    xn_0 = xn_0.reshape(N0, n, d_x)
    xn_0 = tf.convert_to_tensor(xn_0,dtype=tf.float32)

    TT = summary_net(xn_0)
    Diff = tf.reduce_sum((summary_net(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, dtype=tf.float32)
    quan1 = np.quantile(Diff.numpy(), quantile_level) 
    quan1 = min(quan1, epsilon_upper_bound)
    quan1 = tf.constant(quan1, dtype=tf.float32)
    
    
    # -----------------------------
    # Function to Generate Initial MCMC Proposals
    # -----------------------------
    def generate_initial_proposal_mcmc(N_proposal):
        xx_proposal = tf.convert_to_tensor(x_target)
        xx_proposal = tf.tile(xx_proposal, [N_proposal, 1, 1])
        Theta_proposal = amortizer.sample(
            {"summary_conditions": x_target}, n_samples=N_proposal
        )
        Theta_proposal = tf.clip_by_value(Theta_proposal, p_lower, p_upper)
        Theta_proposal = tf.cast(Theta_proposal, "float32")
        return Theta_proposal
    
    # -----------------------------
    # Function to Calculate Prior Density
    # -----------------------------
    def prior(theta):
        """Calculate the prior density of the parameters"""
        mask_ = tf.logical_and(theta >= p_lower, theta <= p_upper)
        prior_ = tf.cast(
            tf.reduce_prod(tf.cast(mask_, "float32"), axis=-1),
            "float32",
        )
        return prior_
    
    def simulate_summary_data(theta, nsims):
        """Generate simulated data from the model and calculate summary statistics"""

        sim_X = np.zeros((theta.shape[0], nsims, n_points, d_x))
        for i_sim in range(theta.shape[0]):
            for j_sim in range(nsims):
                sim_X[i_sim, j_sim] = lotka_volterra_forward(
                    theta[i_sim], n_points, T, x0, y0
                )

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
        kde_ = tf.reduce_mean(tf.exp(- dist / (2 * epsilon**2)), axis=-1)
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
            current_theta, clip_value_min=p_lower, clip_value_max=p_upper
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
    end_refined_time = time.time()
    elapsed_refined_time = end_refined_time - start_refined_time
    elapsed_refined_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_refined_time)
    )

    # save MCMC results
    mcmc_path = os.path.join(bf_ps_folder, f"bf_lv_mcmc_{it}.npy")
    np.save(mcmc_path, Theta_mcmc.numpy())

    # -----------------------------
    # Calculate Bias for MCMC Results
    # -----------------------------

    Theta_mcmc_mean = tf.reduce_mean(Theta_mcmc, axis=0)
    refined_bias = tf.norm(Theta_mcmc_mean - true_ps_tf, ord="euclidean", axis=None)
    refined_bias_vec = tf.abs(Theta_mcmc_mean - true_ps_tf)
    refined_bias_vec = tf.cast(refined_bias_vec, "float32")
       # 绘图
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

    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps_ls):
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


    # 保存图片
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    graph_file = os.path.join(fig_folder, f"bf_lv_refined_experiment_{it}.png")
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
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    )


csv_file = "bf_lv_result1.csv"
credible_interval_file = "bf_lv_credible_interval1.csv"
server = "HPC3"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "server",
            "train_time",
            "refined_time",
            "accp_rate",
            "bias",
            "refined_bias",
        ]
    )

with open(credible_interval_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "ci_lengths_1",
            "ci_lengths_2",
            "ci_lengths_3",
            "ci_lengths_4",
            "ci_lengths_refined_1",
            "ci_lengths_refined_2",
            "ci_lengths_refined_3",
            "ci_lengths_refined_4",
            "low_Y0_1",
            "high_Y0_1",
            "low_Y0_2",
            "high_Y0_2",
            "low_Y0_3",
            "high_Y0_3",
            "low_Y0_4",
            "high_Y0_4",
            "low_Y0_refined_1",
            "high_Y0_refined_1",
            "low_Y0_refined_2",
            "high_Y0_refined_2",
            "low_Y0_refined_3",
            "high_Y0_refined_3",
            "low_Y0_refined_4",
            "high_Y0_refined_4",
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
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    ) = run_experiments(it)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                server,
                elapsed_train_time_str,
                elapsed_refined_time_str,
                accp_rate.numpy(),
                bias.numpy(),
                refined_bias.numpy(),
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
                low_Y0[2],
                high_Y0[2],
                low_Y0[3],
                high_Y0[3],
                low_Y0_refined[0],
                high_Y0_refined[0],
                low_Y0_refined[1],
                high_Y0_refined[1],
                low_Y0_refined[2],
                high_Y0_refined[2],
                low_Y0_refined[3],
                high_Y0_refined[3],
            ]
        )
