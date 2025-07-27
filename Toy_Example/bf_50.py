import os
import csv
import gc
import scipy
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time

# file path
current_dir = os.getcwd()

# parameters for data
N = 12800  #  data_size
n = 50  # sample_size
d = 5  # dimension of parameter theta
d_x = 3  # dimenision of x
p = 10  # dimension of summary statistics

## bayesflow's parameters
batch_size = 256
Np = 500  # number of estimate theta'
epoch = 450
default_lr = 0.0005 

# MCMC Parameters Setup
N_proposal = 50  # 3000
n_samples = 100
burn_in = 2000
thin = 10
Ns = 5
proposed_std = 0.05
quantile_level = 0.0025  # 0.001 # quantile level for bandwidth estimation
epsilon_upper_bound = 0.01

file_path = os.path.join(current_dir, "data", "h_mmd.npy")
h_mmd = np.load(file_path)  # bandwidth of MMD
h_mmd = h_mmd**2

fig_folder = "bf_fig"
os.makedirs(fig_folder, exist_ok=True)

gif_folder = "bf_gif"
os.makedirs(gif_folder, exist_ok=True)

ps_folder = "bf_ps"
os.makedirs(ps_folder, exist_ok=True)

likelihood_bandwidth_path = os.path.join(current_dir, "likelihood_bandwidth_bf.txt")
debug_txt_path = os.path.join(current_dir, "bf_debug.txt")

# class for toy example


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

dtype = np.float32
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


def _Prior(d=4):
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
    x1 = x[0] ** 2 + np.random.normal(loc=0, scale=0.1, size=x[0].shape)
    x = np.insert(x, 1, x1)

    return x[0] if one_sample else x


# compute MMD


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


# bayesflow
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


def run_experiment(it):

    train_start_time = time.time()

    # generate training set
    prior = bf.simulation.Prior(prior_fun=_Prior)

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

    def _simulator(ps, n_obs=200):
        """
        Simulate data at parameters ps.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return _simulator(ps[np.newaxis, :])[0]

        m0, m1, s0, s1, r = _unpack_params(ps)

        us = np.random.randn(n_obs, n, 2)  # standard normal
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1

        xs = stereo_proj(xs)

        return xs

    simulator = bf.simulation.Simulator(simulator_fun=_simulator)

    bayesflow = bf.simulation.GenerativeModel(prior=prior, simulator=simulator)

    summary_net = bf.networks.DeepSet(summary_dim=p)

    inference_net = bf.networks.InvertibleNetwork(num_params=d, num_coupling_layers=5)

    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

    trainer = bf.trainers.Trainer(
        amortizer=amortizer, generative_model=bayesflow, default_lr=default_lr
    )

    it0 = 0
    file_path = os.path.join(current_dir, "data", f"x_train_bf_{it0}.pkl")

    with open(file_path, "rb") as pickle_file:
        offline_data = pickle.load(pickle_file)

    trainer.train_offline(
        offline_data, epochs=epoch, batch_size=batch_size, validation_sims=batch_size
    )

    train_end_time = time.time()

    elapsed_train_time = train_end_time - train_start_time
    elapsed_train_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_train_time))

    # 清除offline data
    del offline_data
    gc.collect()

    x_target = obs_xs.reshape(1, n, 2)
    x_target = stereo_proj(x_target)

    bf_ps = amortizer.sample({"summary_conditions": x_target}, n_samples=Np)


    # save bf_ps
    bf_ps_path = os.path.join(ps_folder, f"bf_{n}_ps_{it}.npy")
    np.save(bf_ps_path, bf_ps)
    
    mmd_bf = compute_mmd(
        tf.cast(bf_ps, "float32"),
        tf.convert_to_tensor(ps, dtype=tf.float32),
        h_mmd,
    )

    # mmd_bf = compute_mmd(
    #     tf.cast(bf_ps[0:10000, :], "float32"),
    #     tf.convert_to_tensor(ps[0:10000, :].astype("float32")),
    #     h_mmd,
    # )

    # 绘图

    sns.set_style("whitegrid")

    true_ps = [1, 1, -1.0, -0.9, 0.6]
    # 定义每个theta_i对应的x轴范围
    x_limits = [
        [0.7, 1.3],  # theta_0
        [0.6, 1.4],  # theta_1
        [-1.5, 1.5],  # theta_2
        [-1.5, 1.5],  # theta_3
        [0, 1.2],  # theta_4
    ]
    # 创建一个图形
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for j, ax in enumerate(axs):
        sns.kdeplot(
            ps[:, j],
            ax=ax,
            fill=False,
            label="posterior",
            color="red",
            linestyle="-.",
            linewidth=1,
        )
        sns.kdeplot(
            bf_ps[:, j],
            ax=ax,
            fill=False,
            label="BF",
            color="blue",
            linestyle="-",
            linewidth=1,
        )

    # 在每个子图上添加竖线表示真实参数的位置
    # for ax, true_p in zip(axs, true_ps):
    #     ax.axvline(true_p, color="r", linestyle="--", linewidth=1)

    # 设置每个子图的标题
    axs[0].set_title("theta1")
    axs[1].set_title("theta2")
    axs[2].set_title("theta3")
    axs[3].set_title("theta4")
    axs[4].set_title("theta5")

    # 保存图片
    plt.legend()
    graph_file = os.path.join(fig_folder, f"bf_{n}_experiment_{it}.png")
    plt.savefig(graph_file)
    plt.close()

    marginal_mmd_list = []
    for j in range(d):
        mmd_marginal = compute_mmd(
            tf.expand_dims(
                tf.cast(bf_ps[:, j], "float32"), axis=1
            ),  # 形状从(N,)变为(N,1)
            tf.expand_dims(tf.convert_to_tensor(ps[:, j].astype("float32")), axis=1),
            1 / n,
        )
        marginal_mmd_list.append(mmd_marginal.numpy())

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

    xn_0 = Stats().calc(Model().sim(Theta0))
    # if xn_0 is tensor, convert to numpy array
    if isinstance(xn_0, tf.Tensor):
        xn_0 = xn_0.numpy()
    xn_0 = xn_0.reshape(N0, n, 2)
    xn_0 = stereo_proj(xn_0)
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)

    TT = summary_net(xn_0)
    Diff = tf.reduce_sum((summary_net(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    quan1 = np.quantile(Diff.numpy(), quantile_level)
    quan1 = min(quan1, epsilon_upper_bound)


    # -----------------------------
    # Create Folders for Saving Figures
    # -----------------------------

    # create a new folder under nn_gif_folder nameed 'bf_gif_{it}'
    temp_gif_folder = os.path.join(gif_folder, f"bf_gif_{it}")
    os.makedirs(temp_gif_folder, exist_ok=True)
    # create d subfolders under temp_gif_folder named 'theta_1', 'theta_2', ..., 'theta_d'

    # create a csv file under temp_gif_folder to record the MMD value for estimated parameters
    mmd_csv_path = os.path.join(temp_gif_folder, f"mmd_mcmc_experiment_{it}.csv")
    with open(mmd_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Step", "MMD(bf, truth)"])
    # record the MMD before MCMC
    with open(mmd_csv_path, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([0, mmd_bf.numpy()])
    for i in range(d):
        theta_i_gif_folder = os.path.join(temp_gif_folder, f"theta_{i+1}")
        os.makedirs(theta_i_gif_folder, exist_ok=True)

    # -----------------------------
    # Plotting Function Definition
    # -----------------------------
    # Plot estimated proposals every 10 steps
    def plot(Theta_seq, ps, bf_ps, true_ps, temp_gif_folder, steps,truncate_window=1):
        print(f"Plotting distributions for step: {steps}")

        # 定义每个theta_i对应的x轴范围
        x_limits = {
            0: [0.7, 1.3],  # theta_1
            1: [0.6, 1.4],  # theta_2
            2: [-1.5, 1.5],  # theta_3
            3: [-1.5, 1.5],  # theta_4
            4: [0, 1.2],  # theta_5
        }

        # 确保Theta_seq有足够的数据
        if len(Theta_seq) < steps - truncate_window:
            print(f"Warning: Not enough data in Theta_seq for step {steps}")
            return

        # Truncate the sequence to the last 1 proposals
        Theta_seq1 = tf.concat(Theta_seq[steps - truncate_window : steps], axis=0)
        Theta_est = Theta_seq1

        # compute the MMD for estimated parameters
        mmd_mcmc = compute_mmd(
            tf.cast(Theta_est[0:10000, :], "float32"),
            tf.convert_to_tensor(ps[0:10000, :].astype("float32")),
            h_mmd,
        )

        for j in range(d):
            # 创建图形
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(6, 4))

            # 确保ps有正确的形状
            if ps.shape[1] <= j:
                print(f"Warning: ps does not have dimension {j}")
                plt.close(fig)
                continue

            # Plot the KDE for the true parameter
            sns.kdeplot(
                ps[:, j],
                ax=ax,
                fill=False,
                label="posterior",
                color="red",
                linestyle="-.",
                linewidth=1,
            )

            # Plot the KDE for the initial proposals
            sns.kdeplot(
                bf_ps[:, j],
                ax=ax,
                fill=False,
                label="BF",
                color="blue",
                linestyle="-",
                linewidth=1,
            )

            # Plot the KDE for the estimated parameters
            sns.kdeplot(
                Theta_est[:, j],
                ax=ax,
                fill=False,
                label="BF+ABC-MCMC",
                color="green",
                linestyle="--",
                linewidth=1,
            )

            # Add vertical line for the true parameter
            # ax.axvline(
            #     true_ps[j], color="red", linestyle="--", linewidth=1, label="True"
            # )

            # 设置x轴范围
            if j in x_limits:
                ax.set_xlim(x_limits[j])

            # Set title and labels
            ax.set_title(f"theta_{j+1} Distribution at Step {steps}")
            ax.set_xlabel(f"theta_{j+1}")
            ax.set_ylabel("Density")
            ax.legend()

            # 调试：显示图形而不立即关闭
            # plt.show()

            # 确保文件夹存在
            theta_j_gif_folder = os.path.join(temp_gif_folder, f"theta_{j+1}")
            os.makedirs(theta_j_gif_folder, exist_ok=True)

            # Save the figure
            print(f"Saving figure for theta_{j+1} at step {steps}")
            graph_path = os.path.join(
                theta_j_gif_folder,
                f"bf_{n}_theta{j+1}_step_{steps}.png",
            )
            plt.savefig(graph_path)

            # 延迟关闭，确保图形已保存
            plt.close(fig)


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
        return Theta_proposal
    
    


    def prior(theta):
        """Calculate the prior density of the parameters"""
        # 选择需要检查的列 [0, 2, 3, 4]
        selected_columns = tf.gather(theta, indices=[0, 2, 3, 4], axis=1)  
        # 检查这些列的绝对值是否小于3，然后相乘
        mask = tf.reduce_prod(tf.where(tf.abs(selected_columns) < 3, 1.0, 0.0), axis=1)
        # 计算高斯部分
        gaussian_part = tf.exp(-((theta[:, 1] - theta[:, 0] ** 2) ** 2) / (2 * 0.1**2))
        # 将结果相乘并转换为float32类型
        prior_ = tf.cast(mask, "float32") * tf.cast(gaussian_part, "float32")
        return prior_

    def simulate_summary_data(theta, nsims):
        """Generate simulated data from the model and calculate summary statistics"""
        sim_X = np.zeros(shape=(theta.shape[0], nsims, n, 3))

        theta_expand = tf.tile(tf.expand_dims(theta, axis=1), [1, nsims, 1])

        for i_sim in range(theta.shape[0]):
            sim_x_ = Stats().calc(Model().sim_preserved_shape(theta_expand[i_sim, :]))
            sim_x_ = stereo_proj(sim_x_)
            sim_X[i_sim] = sim_x_

        TX_ = np.zeros(shape=(theta.shape[0], nsims, p))
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
        # 仅对维度0，2，3，4进行裁剪，使其绝对值小于3
        dim_clip = [0, 2, 3, 4]
        all_dims = tf.range(current_theta.shape[1])  # 列索引：[0,1,2,3,4]
        mask = tf.reduce_any(
            tf.equal(all_dims[:, None], dim_clip), axis=1
        )  # 对每个列判断是否在 dim_clip 中
        current_theta = tf.where(
            mask,  # 广播为 (5,5)，每个元素 (i,j) 的掩码由第 j 列的 mask 决定
            tf.clip_by_value(current_theta, -3, 3),  # 需要裁剪的列：应用裁剪
            current_theta,  # 不需要裁剪的列：保持原样
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
    
    # Save the MCMC results
    ps_mcmc_path = os.path.join(ps_folder, f"bf_{n}_ps_mcmc_{it}.npy")
    np.save(ps_mcmc_path, Theta_mcmc.numpy())

    # -----------------------------
    # Calculate MMD for MCMC Results
    # -----------------------------

    mmd_refinement = compute_mmd(
        tf.cast(Theta_mcmc,"float32"),
        tf.convert_to_tensor(ps, dtype=tf.float32),
        h_mmd,
    )

    # mmd_refinement = compute_mmd(
    #     tf.cast(Theta_mcmc[0:10000, :], "float32"),
    #     tf.convert_to_tensor(ps[0:10000, :].astype("float32")),
    #     h_mmd,
    # )

    # 绘图
    sns.set_style("whitegrid")

    # 创建一个图形
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    # 定义每个theta_i对应的x轴范围
    x_limits = [
        [0.7, 1.3],  # theta_0
        [0.6, 1.4],  # theta_1
        [-1.5, 1.5],  # theta_2
        [-1.5, 1.5],  # theta_3
        [0, 1.2],  # theta_4
    ]

    # 创建一个图形
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for j, ax in enumerate(axs):
        sns.kdeplot(
            ps[:, j],
            ax=ax,
            fill=False,
            label="posterior",
            color="red",
            linestyle="-.",
            linewidth=1,
        )
        sns.kdeplot(
            bf_ps[:, j],
            ax=ax,
            fill=False,
            label="BF",
            color="blue",
            linestyle="-",
            linewidth=1,
        )
        sns.kdeplot(
            Theta_mcmc[:, j],
            ax=ax,
            fill=False,
            label="BF+ABC-MCMC",
            color="green",
            linestyle="--",
            linewidth=1,
        )

    # 在每个子图上添加竖线表示真实参数的位置
    # for ax, true_p in zip(axs, true_ps):
    #     ax.axvline(true_p, color="r", linestyle="--", linewidth=1)

    # 设置每个子图的标题
    axs[0].set_title("theta1")
    axs[1].set_title("theta2")
    axs[2].set_title("theta3")
    axs[3].set_title("theta4")
    axs[4].set_title("theta5")

    # 保存图片
    plt.legend()
    graph_file = os.path.join(fig_folder, f"bf_{n}_refined_experiment_{it}.png")
    plt.savefig(graph_file)
    plt.close()

    refined_marginal_mmd_list = []
    for j in range(d):
        refined_marginal_mmd = compute_mmd(
            tf.expand_dims(
                tf.cast(Theta_mcmc[:, j], "float32"), axis=1
            ),  # 形状从(N,)变为(N,1)
            tf.expand_dims(tf.convert_to_tensor(ps[:, j].astype("float32")), axis=1),
            1 / n,
        )
        refined_marginal_mmd_list.append(refined_marginal_mmd.numpy())

    return (
        elapsed_train_time_str,
        elapsed_refined_time_str,
        mmd_bf,
        mmd_refinement,
        marginal_mmd_list,
        refined_marginal_mmd_list,
        accp_rate.numpy(),
    )


output_file = f"bf_{n}_result1.csv"
marginal_mmd_output_file = f"bf_{n}_marginal_mmd.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment_Index", "train_time", "refined_time", "mmd_bf", "mmd_bf_r", "accp_rate"])

with open(marginal_mmd_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index"]
        + [f"m_mmd theta_{i+1}" for i in range(d)]
        + [f"rm_mmd theta_{i+1}" for i in range(d)]
    )

for i in range(5):
    print(f"Running experiment {i+1}")
    (
        elapsed_train_time_str,
        elapsed_refined_time_str,
        mmd_bf,
        mmd_refinement,
        marginal_mmd_list,
        refined_marginal_mmd_list,
        accp_rate,
    ) = run_experiment(i)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                i + 1,
                elapsed_train_time_str,
                elapsed_refined_time_str,
                mmd_bf.numpy(),
                mmd_refinement.numpy(),
                accp_rate,
            ]
        )

    with open(marginal_mmd_output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i + 1] + marginal_mmd_list + refined_marginal_mmd_list)
