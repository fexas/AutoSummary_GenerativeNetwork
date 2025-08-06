# 修改两个部分
# 0. 新的h_mmd要在data generation的时候就确定
# 1. 计算mmd的batch_size
# 2. 数据载入的方式
import os
from scipy.integrate import odeint
import numpy as np
import tensorflow as tf
import csv
from scipy.integrate import odeint
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings


# file path
current_dir = os.getcwd()
# 创建nn_fig文件夹
fig_folder = "dnnabc_fig"
os.makedirs(fig_folder, exist_ok=True)
ps_folder = "dnnabc_ps"
os.makedirs(ps_folder, exist_ok=True)
quan1_record_csv = "dnn_quan1_record.csv"

# parameters for data
N = 12800  #  data_size
n_points = 200  # n_points 貌似是取代了原来n的位置
n = n_points  # sample_size
d = 4  # dimension of parameter theta
d_x = 2  # dimenision of x
p = d  # dimension of summary statistics
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
rng = np.random

## DNNABC's parameters
default_lr = 0.001
epochs = 700
batch_size = 256
OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}


file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)
x_target = obs_xs.reshape((1, n, d_x))
x_target = tf.convert_to_tensor(x_target, dtype=tf.float32)
dtype = np.float32

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4"]

# lv

def _Prior(batch_size):
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


def simulate_lv_params(
    theta,
    n_points=n_points,
    x0=10,
    y0=5,
    T=15,
):
    """
    Simulates a batch of Ricker datasets given parameters.
    """

    theta = np.atleast_2d(theta)
    X = np.zeros((theta.shape[0], n_points, d_x))

    for j in range(theta.shape[0]):
        X[j] = lotka_volterra_forward(theta[j], n_points, T, x0, y0)

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

# DNN ABC


class DNNABC(keras.Model):
    def __init__(self, T, **kwargs):
        super(DNNABC, self).__init__(**kwargs)
        self.T = T
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        data1 = tf.reshape(data, (batch_size, d + n * d_x))
        Theta = data1[:, :d]
        X = data1[:, d:]
        X = tf.reshape(X, (-1, n, d_x))

        with tf.GradientTape() as tape:

            mse = tf.keras.losses.MeanSquaredError()
            mse_loss = mse(self.T(X), Theta)

        grads = tape.gradient(mse_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(mse_loss)
        return {"loss": self.total_loss_tracker.result()}

    def predict(self, x_obs):
        return self.T(x_obs)

def run_experiments(it):

    # load training data
    file_path = os.path.join(current_dir, "data", "x_train.npy")
    x_train = np.load(file_path)

    # summary network

    num_conv_layers = 2
    lstm_units = 128
    bidirectional = False
    summary_dim = p
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

    dnnabc = DNNABC(T)

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        default_lr, epochs * batch_size, name="lr_decay"
    )

    dnnabc_optimizer = tf.keras.optimizers.Adam(schedule, **OPTIMIZER_DEFAULTS)

    dnnabc.compile(optimizer=dnnabc_optimizer, run_eagerly=False)

    dnnabc.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Determine Threshold
    N_ref = 5000
    Theta_ref = _Prior(N_ref)
    X_ref = simulate_lv_params(Theta_ref,n_points,x0,y0,T_count)
    X_ref = tf.convert_to_tensor(X_ref, dtype=tf.float32)
    TX_ref = dnnabc.T(X_ref)
    TX_to_target_ref = TX_ref - dnnabc.T(x_target)
    TX_to_target_ref = tf.reduce_sum(TX_to_target_ref**2, axis=-1)
    threshold = np.quantile(TX_to_target_ref.numpy(), 0.001)
    with open(quan1_record_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([threshold])

    if Theta_ref.shape[0] == 0:
        warnings.warn("Error: Theta_ref is empty - prior sampling failed")
    elif np.isnan(Theta_ref).any():
        warnings.warn(f"Warning: NaN values found in Theta_ref. Count: {np.isnan(Theta_ref).sum()}")

    if X_ref.shape[0] == 0:
        warnings.warn("Error: X_ref is empty - simulation failed")
    elif np.isnan(X_ref.numpy()).any():
        warnings.warn(f"Warning: NaN values found in X_ref. Count: {np.isnan(X_ref.numpy()).sum()}")

    if np.isnan(TX_ref.numpy()).any():
        warnings.warn(f"Warning: NaN values found in TX_ref. Count: {np.isnan(TX_ref.numpy()).sum()}")
    elif np.isnan(dnnabc.T(x_target).numpy()).any():
        warnings.warn("Warning: NaN values found in dnnabc.T(x_target)")

    if np.isnan(TX_to_target_ref.numpy()).any():
        warnings.warn(f"Warning: NaN values found in TX_to_target_ref. Count: {np.isnan(TX_to_target_ref.numpy()).sum()}")
    
    # ABC_posterior

    N_simulation = 1000
    iter_num = 1000

    for i in range(iter_num):

        Theta_candidate = _Prior(N_simulation)
        X_candidate = simulate_lv_params(
            Theta_candidate, n_points, x0, y0, T_count
        )
        X_candidate = tf.convert_to_tensor(X_candidate, dtype=tf.float32)
        T_X_candidate = dnnabc.T(X_candidate)
        diff = T_X_candidate - dnnabc.T(x_target)
        mse = tf.reduce_sum(diff**2, axis=1)

        if i == 0:
            mse_ = mse
            Theta_candidate_ = Theta_candidate
        else:
            mse_ = tf.concat([mse_, mse], axis=0)
            Theta_candidate_ = tf.concat([Theta_candidate_, Theta_candidate], axis=0)

    idx = tf.where(mse_ < threshold)
    idx = tf.squeeze(idx, axis=1)

    Theta_accp = tf.gather(Theta_candidate_, idx)

    accp_rate = idx.shape[0] / (N_simulation * iter_num)

    # save Theta_accp
    ps_path = os.path.join(ps_folder,f"dnnabc_ps_{it}.npy")
    np.save(ps_path, Theta_accp.numpy())

    Theta_accp_mean = tf.reduce_mean(Theta_accp, axis=0)
    Theta_accp_mean = tf.cast(Theta_accp_mean, dtype=tf.float32)
    bias = tf.norm(Theta_accp_mean - true_ps_tf, ord="euclidean", axis=None)

    # 绘图

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
            Theta_accp[:, j],
            ax=axs[j],
            fill=False,
            label="DNNABC",
            color=est_color,
            linewidth=1.5,
            linestyle="-",
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps_ls):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    # 在每个子图上绘制 95% credible interval
    low, high = credible_interval(Theta_accp)
    ci_length = high - low
    for i in range(4):
        # low, high = credible_interval(Y0)
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color="b", alpha=0.3
        )  # 填充带状区域
        axs[i].axvline(low[i], color=est_color, linestyle="--", linewidth=1.5)
        axs[i].axvline(high[i], color=est_color, linestyle="--", linewidth=1.5)

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(fig_folder, f"dnnabc_lv_{it}.png"))
    plt.close()

    return bias.numpy(), accp_rate, low, high, ci_length


output_file = f"dnnabc_lv_results1.csv"
credible_interval_file = f"dnnabc_lv_credible_interval1.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index", "runtime", "bias", "accpt_rate"]
    )

with open(credible_interval_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "Credible_Interval_Length_Theta1",
            "Credible_Interval_Length_Theta2",
            "Credible_Interval_Length_Theta3",
            "Credible_Interval_Length_Theta4",
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


it = 0

for it in range(10):

    start_time = time.time()
    bias, accp_rate, low, high, ci_length = run_experiments(it)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, elapsed_time_str, bias, accp_rate])

    with open(credible_interval_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                *ci_length,
                low[0],
                high[0],
                low[1],
                high[1],
                low[2],
                high[2],
            ]
        )
