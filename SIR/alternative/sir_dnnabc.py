# 修改两个部分
# 0. 新的h_mmd要在data generation的时候就确定
# 1. 计算mmd的batch_size
# 2. 数据载入的方式
import os
import numpy as np
import tensorflow as tf
import csv
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  LayerNormalization
import time
from scipy.stats import beta

N = 12800
T_steps = 100
n = T_steps
d = 2  # dimension of parameter theta
d_x = 1  # dimenision of x
p = d  # dimension of summary statistics
Q = 1  # number of draw from \exp{\frac{\Vert \theta_i - \theta_j \Vert^2}{w}} in first penalty

## NN's parameters
M = 50  # number of hat_theta_i to estimate MMD
L = 20  # number of unit vector in  S^{d-1} to draw
lambda_1 = 0  # coefficient for 1st penalty
batch_size = 256
OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}

# color setting and upper labels
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels = ["\\theta_1", "\\theta_2"]

# file path
current_dir = os.getcwd()

# 创建nn_fig文件夹
fig_folder = "dnnabc_fig"
os.makedirs(fig_folder, exist_ok=True)
ps_folder = "dnnabc_ps"
os.makedirs(ps_folder, exist_ok=True)


## DNNABC's parameters
default_lr = 0.0005
batch_size = 256
# threshold = 0.25  # threshold for Approximate Bayesian Computation

file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)  # (n, d_x)
x_target = obs_xs.reshape(1, n, d_x)


#  SIR
def prior(batch_size):
    lambda_ = np.random.uniform(0, 1, size=batch_size)  # 感染率
    mu_ = np.random.uniform(0, 1, size=batch_size)  # 恢复率

    theta = np.stack([lambda_, mu_], axis=1)
    return theta


def generate_observation(theta, T_steps):
    # Add dimension check and reshape
    theta = np.atleast_2d(theta)  # 确保输入是二维数组
    lambda_ = theta[:, 0]
    mu_ = theta[:, 1]
    batch_size = theta.shape[0]  # 获取当前批次大小

    # 修改为批量初始化
    I = beta.rvs(1, 100, size=batch_size)  # 添加size参数
    S = 1 - I
    R = np.zeros_like(I)
    sigma = 0.05

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

    return np.array(I_new_obs_list).T


def generate_dataset(theta_candidate, T_steps):
    X = generate_observation(theta_candidate, T_steps)
    X = np.array(X)
    X = X.astype(np.float32)
    return X


def compute_kernel(x, y, h_mmd):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]

    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(-tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2) / (2 * h_mmd))


def compute_mmd(x, y, h_mmd):
    x_kernel = compute_kernel(x, x, h_mmd)
    y_kernel = compute_kernel(y, y, h_mmd)
    xy_kernel = compute_kernel(x, y, h_mmd)
    return (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )

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
    epochs = 700
    default_lr = 0.0005
    true_ps = [0.4, 0.15]
    true_ps_tf = tf.convert_to_tensor(true_ps, dtype=tf.float32)

    # -----------------------------
    # Summary Netowrk
    # -----------------------------
    gru_units = 64
    summary_dim = p

    T = GRUSummary(gru_units=gru_units, summary_dim=summary_dim, dropout_rate=0.1)


    dnnabc = DNNABC(T)

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        default_lr, epochs * batch_size, name="lr_decay"
    )

    dnnabc_optimizer = tf.keras.optimizers.Adam(schedule, **OPTIMIZER_DEFAULTS)

    dnnabc.compile(optimizer=dnnabc_optimizer, run_eagerly=False)

    # load training data
    file_path = os.path.join(current_dir, "data", "x_train.npy")
    x_train = np.load(file_path)

    dnnabc.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # ABC_posterior

    ## determine threshold
    N_ref = 5000
    Theta_ref = prior(N_ref)
    X_ref = generate_dataset(Theta_ref, T_steps)
    X_ref = tf.convert_to_tensor(X_ref, dtype=tf.float32)
    X_ref = tf.reshape(X_ref, (N_ref, n, d_x))
    TX_ref = dnnabc.predict(X_ref)
    TX_diff_ref = TX_ref - dnnabc.predict(x_target)
    TX_diff_ref = tf.reduce_sum(TX_diff_ref**2, axis=-1)
    threshold = np.quantile(TX_diff_ref.numpy(),0.001)

    N_simulation = 1000
    iter_num = 500

    for i in range(iter_num):

        Theta_candidate = prior(N_simulation)
        X_candidate = generate_dataset(Theta_candidate, T_steps)
        Theta_candidate = tf.convert_to_tensor(Theta_candidate, dtype=tf.float32)
        X_candidate = tf.convert_to_tensor(X_candidate, dtype=tf.float32)
        X_candidate = tf.reshape(X_candidate, (N_simulation, n, d_x))

        T_X_candidate = dnnabc.predict(X_candidate)
        diff = T_X_candidate - dnnabc.predict(x_target)
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

    # save result
    ps_path = os.path.join(ps_folder,f"sir_dnnabc_{it}.npy")
    np.save(ps_path, Theta_accp.numpy())

    # calculate bias
    true_ps_tf = tf.convert_to_tensor(true_ps, dtype=tf.float32)
    Theta_accp_mean = tf.reduce_mean(Theta_accp, axis=0)
    bias = tf.norm(Theta_accp_mean - true_ps_tf, ord="euclidean", axis=None)

    bias_vec = tf.abs(Theta_accp_mean - true_ps_tf)
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

    # 绘图

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

    for ax, true_p in zip(axs, true_ps):
        ax.axvline(true_p, color=truth_color, linestyle="-", linewidth=1.5)

    low, high = credible_interval(Theta_accp)
    ci_length = high - low
    for i in range(d):
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
    plt.savefig(os.path.join(fig_folder, f"sir_dnnabc_{it}.png"))
    plt.close()

    return accp_rate, bias, bias_vec, low, high, ci_length


output_file = f"sir_dnnabc_result1.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "runtime",
            "accpt_rate",
            "bias",
            "bias_1",
            "bias_2",
            "ci_length_1",
            "ci_length_2",
            "low_1",
            "high_1",
            "low_2",
            "high_2",
        ]
    )


for it in range(10):

    start_time = time.time()
    accp_rate, bias, bias_vec, low, high, ci_length = run_experiments(it)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                it,
                elapsed_time_str,
                accp_rate,
                bias.numpy(),
                bias_vec[0].numpy(),
                bias_vec[1].numpy(),
                *ci_length,
                low[0],
                high[0],
                low[1],
                high[1],
            ]
        )
