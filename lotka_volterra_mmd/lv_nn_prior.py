# remark
# 1. bandwidth of MMD for lotka-volterra needs to be large enough
import os
import gc
import numpy as np
import tensorflow as tf
import math
import csv
from scipy.integrate import odeint
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

# activate gpu
# 设置GPU内存增长，避免显存被占满
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        # 设置GPU内存自动增长
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"成功启用GPU: {physical_devices}")
    except RuntimeError as e:
        print(e)

# 验证GPU是否可用
print("GPU可用状态:", tf.test.is_gpu_available())
print("TensorFlow版本:", tf.__version__)

N = 12800
n_points = 200  # n_points 貌似是取代了原来n的位置
n = n_points  # sample_size
d = 4  # dimension of parameter theta
d_x = 2  # dimenision of x
p = 9  # dimension of summary statistics
T_count = 15
x0 = 10
y0 = 5
p_lower = -5
p_upper = 2

## NN's parameters
batch_size = 256
M = 50  # number of hat_theta_i to estimate MMD
n_samples_z = 20  # number of unit vector in  S^{d-1} to draw
default_lr = 0.0005
epochs = 700
Np = 500  # number of estimate theta

# MCMC Parameters Setup
Ns = 5  # 5
N_proposal = 500  # 3000
burn_in = 249
n_samples = 1
thin = 10
proposed_std = 0.05  # 1 / (2 * math.sqrt(n))
quantile_level = 0.001
epsilon_upper_bound = 0.035
# quantile_level_list = [0.05, 0.025, 0.01, 0.0075, 0.005, 0.001]

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4"]

# file path
current_dir = os.getcwd()

fig_folder = "nn_prior_fig"
os.makedirs(fig_folder, exist_ok=True)

ps_folder = "nn_ps"
os.makedirs(ps_folder, exist_ok=True)

quan1_record_csv = "quan1_record.csv"

# load data
rng = np.random
true_ps = np.log(np.array([1, 0.01, 0.5, 0.01]))
true_ps_tf = tf.convert_to_tensor(true_ps, dtype=tf.float32)
true_ps_ls = true_ps.tolist()

dtype = np.float32
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)  # (n, d_x)
x_target = obs_xs.reshape(1, n_points, d_x)


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


def simulate_lv_params(
    theta,
    n_points=n_points,
    x0=10,
    y0=5,
    T=15,
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


# -----------------------------
# MultiConv1D
# -----------------------------


class MultiConv1D(tf.keras.layers.Layer):  # 改为继承Layer而不是Model
    """Implements an inception-inspired 1D convolutional layer using different kernel sizes."""

    def __init__(self, settings):
        """Creates an inception-like Conv1D layer

        Parameters
        ----------
        settings  : dict
            A dictionary which holds the arguments for the internal ``Conv1D`` layers.
        """
        super().__init__()  # 移除**kwargs参数
        self.settings = settings
        # 延迟创建卷积层，移至build方法
        self.convs = None
        self.dim_red = None

    def build(self, input_shape):
        # 在build方法中创建子层，确保输入形状已知
        self.convs = [
            Conv1D(kernel_size=f, **self.settings["layer_args"])
            for f in range(
                self.settings["min_kernel_size"], self.settings["max_kernel_size"]
            )
        ]

        # 创建最终的Conv1D层用于降维
        dim_red_args = {
            k: v
            for k, v in self.settings["layer_args"].items()
            if k not in ["kernel_size", "strides"]
        }
        dim_red_args["kernel_size"] = 1
        dim_red_args["strides"] = 1
        self.dim_red = Conv1D(**dim_red_args)
        super().build(input_shape)  # 标记层为已构建

    def call(self, x):  # 移除**kwargs参数
        """Performs a forward pass through the layer.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, n_time_steps, n_time_series)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, n_time_steps, n_filters)
        """
        out = self._multi_conv(x)
        out = self.dim_red(out)
        return out

    def _multi_conv(self, x):  # 移除**kwargs参数
        """Applies the convolutions with different sizes and concatenates outputs."""
        return tf.concat([conv(x) for conv in self.convs], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"settings": self.settings})
        return config


# -----------------------------
# NN
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

        bandwidth = tf.constant([50 / n], "float32")
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

        mmd = tf.reduce_mean(K_gg) * M / (M - 1) - 2 * tf.reduce_mean(K_gt)

        return mmd

    def SliceMMD(self, theta, Gtheta):
        """
        Args:
        theta: [batch_size,d]
        Gtheta: [batch_size,M,d]

        """

        bandwidth = tf.constant([1 / (2 * n)], "float32")
        constant = tf.sqrt(1 / bandwidth)

        unit_vectors = tf.random.normal(shape=(n_samples_z, d))
        unit_vectors_norm = tf.norm(unit_vectors, axis=1, keepdims=True)
        unit_vectors = unit_vectors / unit_vectors_norm  # (L,d)
        unit_vectors = tf.transpose(unit_vectors, perm=[1, 0])  # (d,L)

        Gtheta_diff = tf.expand_dims(Gtheta, 1) - tf.expand_dims(
            Gtheta, 2
        )  # (N, M, M, d)
        slice_Gtheta_diff = tf.matmul(Gtheta_diff, unit_vectors)  # (N,M,M,L)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5 * tf.square(slice_Gtheta_diff) / bandwidth[:, None, None, None, None]
        )

        loss_term1 = tf.reduce_mean(marginal_p)

        # second_summation
        theta_minus_Gtheta = tf.expand_dims(theta, 1) - Gtheta  # （N,M,d）
        slice_theta_minus_Gtheta = tf.matmul(
            theta_minus_Gtheta, unit_vectors
        )  # (N,M,L)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5
            * tf.square(slice_theta_minus_Gtheta)
            / bandwidth[:, None, None, None, None]
        )
        loss_term2 = tf.reduce_mean(marginal_p)

        slice_MMD_loss = loss_term1 * M /(M-1) - 2 * loss_term2

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
            loss = self.MMD(theta=Theta_, Gtheta=G_theta)  # 仅在tape内计算损失

        grads = tape.gradient(loss, self.trainable_weights)

        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)

        return {"loss": self.total_loss_tracker.result()}


def run_experiments(it):

    # summary network

    num_conv_layers = 2
    lstm_units = 128
    bidirectional = False
    summary_dim = 9
    # conv_settings = defaults.DEFAULT_SETTING_MULTI_CONV
    conv_settings = {
        "layer_args": {
            "activation": "relu",
            "filters": 32,
            "strides": 1,
            "padding": "causal",
        },
        "min_kernel_size": 1,
        "max_kernel_size": 3,
    }

    net = Sequential([MultiConv1D(conv_settings) for _ in range(num_conv_layers)])
    lstm = Bidirectional(LSTM(lstm_units)) if bidirectional else LSTM(lstm_units)
    out_layer = Dense(summary_dim, activation="linear")

    T_inputs = keras.Input(shape=([n, d_x]))
    x = net(T_inputs)
    x1 = lstm(x)
    T_outputs = out_layer(x1)

    T = keras.Model(inputs=T_inputs, outputs=T_outputs)
    T.summary()

    # generator network
    intermediate_dim_G = 256
    G_inputs = keras.Input(shape=([p + d]))  # Input: z_i & T(x_{1:n}^i)

    x = layers.Dense(
        units=intermediate_dim_G,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(G_inputs)
    x = layers.Dense(
        units=128,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
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
    nn.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[loss_history])
    train_time_end = time.time()

    elapsed_train_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(train_time_end - train_time_start)
    )

    # 释放x_train
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

    # 保存预测的theta
    ps_path = os.path.join(ps_folder, f"lv_nn_ps_{it}.npy")
    np.save(ps_path, Y0.numpy())

    # caluate bias
    Y0_mean = tf.reduce_mean(Y0, axis=0)
    bias = tf.norm(Y0_mean - true_ps_tf, ord="euclidean", axis=None)

    # construct 95% credible interval for each parameter with empirical value Y0

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
        [-1.0, -0.5],  # theta_2
        [-5.2, -4.0],  # theta_3
    ]
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            Y0[:, j],
            ax=axs[j],
            label="MMD",
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
    low, high = credible_interval(Y0)
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
    graph_path = os.path.join(fig_folder, f"lv_nn_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    # -----------------------------
    # MCMC Refinement Overview
    # -----------------------------
    # Refinement using Monte Carlo ABC with weight being calculated as a kernel regression estimator or direct sample estimation
    # This section implements MCMC to refine the parameter estimation results.
    
    TX_target_ = nn.T(x_target)

    # -----------------------------
    # Calculate Bandwidth for Likelihood Estimator
    # -----------------------------
    N0 = 5000
    xx = tf.convert_to_tensor(x_target)
    xx = tf.tile(xx, [N0, 1, 1])
    x = nn.T(xx)
    Y = tf.random.normal(shape=[N0, d])
    YY = tf.concat([Y, x], 1)
    Theta0 = nn.G(YY)
    Theta0 = tf.cast(Theta0, "float32")

    xn_0 = simulate_lv_params(
        Theta0.numpy(), n_points, x0, y0, T_count, to_tensor=False
    )
    xn_0 = xn_0.reshape(N0, n, d_x)
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)

    TT = nn.T(xn_0)
    Diff = tf.reduce_sum((T(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, dtype=tf.float32)
    quan1 = np.quantile(Diff.numpy(), quantile_level)
    # record the quan1 value in a CSV file
    with open(quan1_record_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([quan1])
    quan1 = min(quan1, epsilon_upper_bound)
    quan1 = tf.constant(quan1, dtype=tf.float32)

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
                    theta[i_sim], n_points, T_count, x0, y0
                )

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
        acceptance_rate = accepted / ((n_samples+burn_in) * N_proposal)
        samples = tf.concat(samples, axis=0)

        return samples, acceptance_rate

    # -----------------------------
    # Run MCMC Refinement
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
    mcmc_path = os.path.join(ps_folder, f"lv_nn_mcmc_{it}.npy")
    np.save(mcmc_path, Theta_mcmc.numpy())

    # -----------------------------
    # Calculate bias for MCMC Results
    # -----------------------------    # caluate bias
    Theta_mcmc_mean = tf.reduce_mean(Theta_mcmc, axis=0)
    refined_bias = tf.norm(Theta_mcmc_mean - true_ps_tf, ord="euclidean", axis=None)

    # -----------------------------
    # Plot Final Posterior Estimation
    # -----------------------------

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    x_limits = [
        [-0.4, 0.4],  # theta_0
        [-5.2, -4.0],  # theta_1
        [-1.0, -0.5],  # theta_2
        [-5.2, -4.0],  # theta_3
    ]
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            Y0[:,j],
            ax=axs[j],
            fill=False,
            label="MMD",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        sns.kdeplot(
            Theta_mcmc[:, j],
            ax=axs[j],
            fill=False,
            label="MMD+ABC-MCMC",
            color=refined_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")


    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps_ls):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    low_Y0, high_Y0 = credible_interval(Y0)
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
    graph_path = os.path.join(fig_folder, f"lv_nn_refined_{it}.png")
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
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    )


csv_file = "nn_prior_lv_result1.csv"
credible_interval_file = "nn_prior_lv_credible_interval1.csv"


with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "train_time",
            "refined_time",
            "bias",
            "refined_bias",
            "accp_rate",
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

    print(f"Running experiment {it + 1}...")
    (
        train_time_str,
        refined_time_str,
        bias,
        refined_bias,
        low_Y0,
        high_Y0,
        low_Y0_refined,
        high_Y0_refined,
        accp_rate,
        ci_lengths,
        ci_lengths_refined,
    ) = run_experiments(it)
    print(f"Experiment {it + 1} completed.")

    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                it,
                train_time_str,
                refined_time_str,
                bias.numpy(),
                refined_bias.numpy(),
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
