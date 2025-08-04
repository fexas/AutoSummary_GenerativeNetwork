# 修改两个部分
# 0. 新的h_mmd要在data generation的时候就确定
# 1. 计算mmd的batch_size
# 2. 数据载入的方式
import os
import scipy
import numpy as np
import tensorflow as tf
import csv
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import time

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
n = 25  # sample_size
d = 5  # dimension of parameter theta
d_x = 3  # dimenision of x
p = 5  # dimension of summary statistics
file_path = os.path.join(current_dir, "data", "h_mmd.npy")
h_mmd = np.load(file_path)  # bandwidth of MMD
h_mmd = h_mmd**2
OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}

## DNNABC's parameters
default_lr = 0.0005
epochs = 500
batch_size = 256

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4","\\theta_5"]

# preliminary class

class toy_prior:
    def __init__(self):
        pass

    def random(self, batch_size):
        """Generate samples from the prior distribution for toy model parameters."""
        theta = np.random.uniform(-3, 3, size=(batch_size, d))
        theta[:, 1] = theta[:, 0] ** 2 + np.random.randn(batch_size) * 0.1
        return theta

class Uniform:
    """
    Parent class for uniform distributions.
    """

    def __init__(self, n_dims):

        self.n_dims = n_dims

    def grad_log_p(self, x):
        """
        :param x: rows are datapoints
        :return: d/dx log p(x)
        """

        x = np.asarray(x)
        assert (x.ndim == 1 and x.size == self.n_dims) or (
            x.ndim == 2 and x.shape[1] == self.n_dims
        ), "wrong size"

        return np.zeros_like(x)


class BoxUniform(Uniform):
    """
    Implements a uniform pdf, constrained in a box.
    """

    def __init__(self, lower, upper):
        """
        :param lower: array with lower limits
        :param upper: array with upper limits
        """

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        assert (
            lower.ndim == 1 and upper.ndim == 1 and lower.size == upper.size
        ), "wrong sizes"
        assert np.all(lower < upper), "invalid upper and lower limits"

        Uniform.__init__(self, lower.size)

        self.lower = lower
        self.upper = upper
        self.volume = np.prod(upper - lower)

    def eval(self, x, ii=None, log=True):
        """
        :param x: evaluate at rows
        :param ii: a list of indices to evaluate marginal, if None then evaluates joint
        :param log: whether to return the log prob
        :return: the prob at x rows
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:

            in_box = np.logical_and(self.lower <= x, x <= self.upper)
            in_box = np.logical_and.reduce(in_box, axis=1)

            if log:
                prob = -float("inf") * np.ones(in_box.size, dtype=float)
                prob[in_box] = -np.log(self.volume)

            else:
                prob = np.zeros(in_box.size, dtype=float)
                prob[in_box] = 1.0 / self.volume

            return prob

        else:
            assert len(ii) > 0, "list of indices can" "t be empty"
            marginal = BoxUniform(self.lower[ii], self.upper[ii])
            return marginal.eval(x, None, log)

    def gen(self, n_samples=None, rng=np.random):
        """
        :param n_samples: int, number of samples to generate
        :return: numpy array, rows are samples. Only 1 sample (vector) if None
        """

        one_sample = n_samples is None
        u = rng.rand(1 if one_sample else n_samples, self.n_dims)
        x = (self.upper - self.lower) * u + self.lower

        return x[0] if one_sample else x


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


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    est_ps = [1, 1, -1.0, -0.9, 0.6]

    rng = np.random.RandomState()
    obs_xs = Stats().calc(Model().sim(est_ps, rng=rng))

    return est_ps, obs_xs


# real posterior
rng = np.random
true_ps, _ = get_ground_truth()
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)
dtype = np.float32

file_path = os.path.join(current_dir, "data", "ps.npy")
ps = np.load(file_path)

# class for data  generation and data trasformation (training datasets)


def stereo_proj(A):
    X_comp = A[..., 0]
    Y_comp = A[..., 1]

    # 计算新的分量
    new_X_comp = 2 * X_comp / (1 + X_comp**2 + Y_comp**2)
    new_Y_comp = 2 * Y_comp / (1 + X_comp**2 + Y_comp**2)
    Z_comp = (-1 + X_comp**2 + Y_comp**2) / (1 + X_comp**2 + Y_comp**2)

    result = np.stack([new_X_comp, new_Y_comp, Z_comp], axis=-1)

    return result


def _Prior(d=5):
    """
    :param n_samples: int, number of samples to generate
    :return: numpy array, rows are samples. Only 1 sample (vector) if None
    """
    lower = [-3.0] * d
    upper = [+3.0] * d
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    one_sample = d is None
    u = np.random.rand(d)
    x = (upper - lower) * u + lower
    a = np.random.randn(1)
    x[1] = x[0] ** 2 + a[0] * 0.1
    return x[0] if one_sample else x


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
# NN Module
# -----------------------------


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""

    pass


class InvariantModule(tf.keras.Model):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create internal functions
        self.s1 = Sequential(
            [
                Dense(**settings["dense_s1_args"])
                for _ in range(settings["num_dense_s1"])
            ]
        )
        self.s2 = Sequential(
            [
                Dense(**settings["dense_s2_args"])
                for _ in range(settings["num_dense_s2"])
            ]
        )

        # Pick pooling function
        if settings["pooling_fun"] == "mean":
            pooling_fun = partial(tf.reduce_mean, axis=-2)
        elif settings["pooling_fun"] == "max":
            pooling_fun = partial(tf.reduce_max, axis=-2)
        else:
            if callable(settings["pooling_fun"]):
                pooling_fun = settings["pooling_fun"]
            else:
                raise ConfigurationError("pooling_fun argument not understood!")
        self.pooler = pooling_fun

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size,..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        x_reduced = self.pooler(self.s1(x, **kwargs))
        out = self.s2(x_reduced, **kwargs)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an equivariant module according to [1] which combines equivariant transforms
        with nested invariant transforms, thereby enabling interactions between set members.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` constructor.
        """

        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential(
            [
                Dense(**settings["dense_s3_args"])
                for _ in range(settings["num_dense_s3"])
            ]
        )

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, ..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, ..., equiv_dim)
        """

        # Store shape of x, will be (batch_size, ..., some_dim)
        shape = tf.shape(x)

        # Example: Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x, **kwargs)
        out_inv = tf.expand_dims(out_inv, -2)
        tiler = [1] * len(shape)
        tiler[-2] = shape[-2]
        out_inv_rep = tf.tile(out_inv, tiler)

        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)

        # Pass through equivariant func
        out = self.s3(out_c, **kwargs)
        return out


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


x_target = obs_xs.reshape((1, n, 2))
x_target = stereo_proj(x_target)
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

    file_path = os.path.join(current_dir, "data", "h_mmd.npy")
    h_mmd = np.load(file_path)  # bandwidth of MMD
    h_mmd = h_mmd**2

    model = Model()

    # load training data
    it0 = 0
    file_path = os.path.join(current_dir, "data", "x_train_%d.npy" % it0)
    x_train = np.load(file_path)

    ## summary_net -- deep set
    settings = dict(
        num_dense_s1=2,
        num_dense_s2=2,
        num_dense_s3=2,
        dense_s1_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        dense_s2_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        dense_s3_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        pooling_fun="mean",
    )

    summary_dim = p

    num_equiv = 2
    equiv_layers = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
    inv = InvariantModule(settings)
    out_layer = layers.Dense(summary_dim, activation="linear")
    T_inputs = keras.Input(shape=([n, d_x]))
    x = equiv_layers(T_inputs)
    T_outputs = out_layer(inv(x))
    T = keras.Model(T_inputs, T_outputs, name="T")
    T.summary()

    dnnabc = DNNABC(T)

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        default_lr, epochs * batch_size, name="lr_decay"
    )

    dnnabc_optimizer = tf.keras.optimizers.Adam(schedule, **OPTIMIZER_DEFAULTS)

    dnnabc.compile(optimizer=dnnabc_optimizer, run_eagerly=False)

    dnnabc.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # determine ABC threshold
    N_ref = 5000
    Theta_ref = toy_prior().random(N_ref)
    X_ref = model.sim_preserved_shape(ps=Theta_ref)
    X_ref = stereo_proj(X_ref)
    X_ref = tf.convert_to_tensor(X_ref, dtype=tf.float32)
    X_ref = tf.reshape(X_ref, (N_ref, n, d_x))
    TX_ref = dnnabc.predict(X_ref)
    TX_to_target_ref = TX_ref -dnnabc.predict(x_target)
    TX_to_target_ref = tf.reduce_sum(TX_to_target_ref**2, axis=-1)
    threshold = np.quantile(TX_to_target_ref.numpy(), 0.001)
    with open(quan1_record_csv, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([threshold])

    # ABC_posterior
    
    N_simulation = 1000
    iter_num = 5000
    Theta_accp = []
    accp_num = 0

    for i in range(iter_num):

        Theta_candidate = toy_prior().random(N_simulation)
        X_candidate = model.sim_preserved_shape(ps=Theta_candidate)
        X_candidate = stereo_proj(X_candidate)
        Theta_candidate = tf.convert_to_tensor(Theta_candidate, dtype=tf.float32)
        X_candidate = tf.convert_to_tensor(X_candidate, dtype=tf.float32)
        X_candidate = tf.reshape(X_candidate, (N_simulation, n, d_x))

        T_X_candidate = dnnabc.predict(X_candidate)
        diff = T_X_candidate - dnnabc.predict(x_target)
        mse = tf.reduce_sum(diff**2, axis=1)

        idx = tf.where(mse < threshold)
        idx = tf.squeeze(idx, axis=1)

        Theta_candidate_accp = tf.gather(Theta_candidate, idx)
        Theta_accp.append(Theta_candidate_accp.numpy())
        accp_num += len(idx)

    Theta_accp = np.concatenate(Theta_accp, axis=0)
    Theta_accp = tf.convert_to_tensor(Theta_accp, dtype=tf.float32)
    accp_rate = accp_num / (N_simulation * iter_num)

    ps_tf = tf.convert_to_tensor(ps, dtype=tf.float32)
    mmd_dnnabc = compute_mmd(Theta_accp, ps_tf, h_mmd)

    # save Theta_accp
    ps_path = os.path.join(ps_folder, f"dnnabc_ps_{it}.npy")
    np.save(ps_path, Theta_accp.numpy())

    # 绘图

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
            color=truth_color,
            label="posterior",
            linewidth=1.5,
            linestyle="-.",
        )
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

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(fig_folder, f"DNNABC_{it}.png"))
    plt.close()

    return accp_rate, mmd_dnnabc.numpy()


output_file = f"dnnabc_{n}_result1.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index", "runtime", "accpt_rate", "mmd"]
    )


for it in range(10):
    start_time = time.time()
    accp_rate, mmd_dnnabc = run_experiments(it)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([it, elapsed_time_str, accp_rate, mmd_dnnabc])
