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
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Conv1D
import seaborn as sns
import matplotlib.pyplot as plt
import time


# file path
current_dir = os.getcwd()
# 创建nn_fig文件夹
fig_folder = "dnnabc_fig"
os.makedirs(fig_folder, exist_ok=True)

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
p_lower = -2
p_upper = 2


## DNNABC's parameters
default_lr = 0.0005
batch_size = 256
Np = 20000  # number of estimate theta
threshold = 0.25  # threshold for Approximate Bayesian Computation
OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}

rng = np.random
true_ps = np.array([1, 1, 1, 1])
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)
dtype = np.float32


# lv

p_lower_list = np.array([-2, -2, -2, -2])
p_higher_list = np.array([2, 2, 2, 2])


def _Prior(batch_size):
    theta = np.random.uniform(p_lower_list, p_higher_list, size=(batch_size, 4))
    return np.exp(theta)


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
    t_steps = 400
    t_span = [0, T_span]
    alpha, beta, gamma, delta = np.exp(params)
    initial_state = [x0, y0]
    noise_scale = 0.1

    x, y, t = ecology_model(
        alpha,
        beta,
        gamma,
        delta,
        t_span=t_span,
        t_steps=t_steps,
        initial_state=initial_state,
    )

    # Add Gaussian noise to observations
    noisy_x = rng.normal(x, noise_scale)
    noisy_y = rng.normal(y, noise_scale)

    step_indices = np.arange(0, t_steps, 1)
    observed_indices = np.sort(rng.choice(step_indices, n_obs, replace=False))

    # concatenate the observed time series of x and y
    observed_X = np.concatenate(
        (
            noisy_x[observed_indices].reshape(-1, 1),
            noisy_y[observed_indices].reshape(-1, 1),
        ),
        axis=1,
    )

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
    X = np.zeros((theta.shape[0], n_points, 2))

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


x_target = obs_xs.reshape((1, n, d_x))
x_target = tf.convert_to_tensor(x_target, dtype=tf.float32)


def compute_abc_metric(dnnabc, x):

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = tf.reshape(x, (1, n, d_x))

    S_x = dnnabc.predict(x)
    S_x_obs = dnnabc.predict(x_target)

    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(S_x, S_x_obs)

    return mse_loss


def run_experiments(it):

    batch_size = 256
    epochs = 450
    default_lr = 0.0005

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

    dnnabc.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # ABC_posterior

    # thresold = 0.25
    N_simulation = 1000
    iter_num = 10

    for i in range(iter_num):

        Theta_candidate = _Prior(N_simulation)
        X_candidate = simulate_lv_params(
            Theta_candidate, n_points, x0, y0, T_count
        )

        T_X_candidate = dnnabc.predict(X_candidate)
        diff = T_X_candidate - dnnabc.predict(x_target)
        mse = tf.reduce_sum(diff**2, axis=1)

        if i == 0:
            mse_ = mse
            Theta_candidate_ = Theta_candidate
        else:
            mse_ = tf.concat([mse_, mse], axis=0)
            Theta_candidate_ = tf.concat([Theta_candidate_, Theta_candidate], axis=0)

    # 测试一下仅接受前1%的样本拟合出来是什么结果
    mse_threshold = tf.sort(mse_)[int(0.15 * N_simulation * iter_num)]
    threshold = np.array(mse_threshold)

    idx = tf.where(mse_ < threshold)
    idx = tf.squeeze(idx, axis=1)

    Theta_accp = tf.gather(Theta_candidate_, idx)
    print(Theta_accp.shape)

    accp_rate = idx.shape[0] / N_simulation

    Theta_accp_mean = tf.reduce_mean(Theta_accp, axis=0)
    Theta_accp_mean = tf.cast(Theta_accp_mean, dtype=tf.float32)
    true_ps_tf = tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
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

    true_ps = [1, 1, 1, 1]

    # 创建一个图形
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    # 为每个分量绘制 KDE 图

    for j in range(d):
        sns.kdeplot(Theta_accp[:, j], ax=axs[j], fill=False, label="DNNABC",color='blue')

    # 在每个子图上添加竖线表示真实参数的位置
    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color="r", linestyle="--", linewidth=1)

    # 在每个子图上绘制 95% credible interval
    low, high = credible_interval(Theta_accp)
    ci_length = high - low
    for i in range(4):
        # low, high = credible_interval(Y0)
        axs[i].fill_betweenx(
            axs[i].get_ylim(), low[i], high[i], color="b", alpha=0.3
        )  # 填充带状区域
        axs[i].axvline(low[i], color="b", linestyle="--", linewidth=1)
        axs[i].axvline(high[i], color="b", linestyle="--", linewidth=1)

    # 设置每个子图的标题
    axs[0].set_title("theta1")
    axs[1].set_title("theta2")
    axs[2].set_title("theta3")
    axs[3].set_title("theta4")

    # 保存图片
    plt.legend()
    plt.savefig(os.path.join(fig_folder, f"dnnabc_lv_{it}.png"))
    plt.close()

    return bias.numpy(), accp_rate, mse_threshold, low, high, ci_length


output_file = f"dnnabc_lv_results1.csv"
credible_interval_file = f"dnnabc_lv_credible_interval1.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index", "runtime", "bias", "accpt_rate", "0.001 threshold"]
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
    bias, accp_rate, mse_threshold, low, high, ci_length = run_experiments(it)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, elapsed_time_str, bias, accp_rate, mse_threshold.numpy()])

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
