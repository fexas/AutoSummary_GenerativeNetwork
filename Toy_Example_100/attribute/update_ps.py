import os
import scipy
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# parameters for data
N = 12800  #  data_size
n = 100  # sample_size
d = 5  # dimension of parameter theta
d_x = 3  # dimenision of x
p = 10  # dimension of summary statistics

current_dir = os.getcwd()

# 创建data文件夹
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

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
true_ps = [1, 1, -1.0, -0.9, 0.6]
file_path = os.path.join(current_dir, "data", "obs_xs.npy")
obs_xs = np.load(file_path)

#  0612 wait to check
obs_xs = np.reshape(obs_xs, (n, 2))

dtype = np.float32


def log_posterior(theta):
    if (
        abs(theta[0]) >= 3
        or abs(theta[2]) >= 3
        or abs(theta[3]) >= 3
        or abs(theta[4]) >= 3
    ):
        return -np.inf
    else:
        u = [theta[0], theta[1]]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rho = math.tanh(theta[4])

        Sigma = [
            [s1**2, rho * s1 * s2],
            [rho * s1 * s2, s2**2],
        ]

        IS = np.linalg.inv(Sigma)
        quad_form = np.matrix.trace(
            np.matmul(np.matmul(IS, (obs_xs - u).T), (obs_xs - u))
        )
        log_likelihood = -0.5 * quad_form - 0.5 * n * np.log(np.linalg.det(Sigma))
        log_prior = -((theta[1] - theta[0] ** 2) ** 2) / (2 * 0.1**2)
        return log_likelihood + log_prior
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


# MCMC sampling
# running N_proposal chain simutaneously
# theta_intial_proposal

proposed_std = 0.05 # 0.05 # 1 / math.sqrt(n) # 0.15 / math.sqrt(n) * math.sqrt(10)
N_proposal = 1000 # 5000
n_samples = 150 # 100
burn_in_steps = 5000 # 5000 # 3000 # 2500
thin = 30 # 20
prior = bf.simulation.Prior(prior_fun=_Prior)

# -----------------------------
# Function to Generate Initial MCMC Proposals
# -----------------------------
def generate_initial_proposal_mcmc(N_proposal):
    return prior(N_proposal)["prior_draws"]

# -----------------------------
# Function to Calculate log posterior for an array of parameters
# -----------------------------
def log_posterior_array(theta):
    """
    Calculate the log posterior for an array of parameters.
    :param theta: numpy array of shape (n_samples, d)
    :return: numpy array of log posterior values
    """
    log_posteriors = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        log_posteriors[i] = log_posterior(theta[i])
    return log_posteriors

# -----------------------------
# MCMC 
# -----------------------------

def mcmc(N_proposal, burn_in_steps, n_samples, thin):
    """Run N_proposal MCMC chains simultaneously."""
    Theta_seq = []
    accp = 0

    Theta_proposal = generate_initial_proposal_mcmc(N_proposal)
    log_posterior_0 = log_posterior_array(Theta_proposal)

    for mcmc_step in range(burn_in_steps + n_samples):
        # generate new proposal using numpy
        Theta_new_proposal = np.random.normal(
            loc=Theta_proposal, scale=proposed_std, size=(N_proposal, 5)
        )
        # calculate log accptance ratio
        log_posterior_1 = log_posterior_array(Theta_new_proposal)
        log_ratio = log_posterior_1 - log_posterior_0
        u = np.log(np.random.uniform(size=N_proposal))
        # acceptance condition
        accept = u <= log_ratio

        # update Theta_proposal
        Theta_proposal[accept] = Theta_new_proposal[accept]
        log_posterior_0[accept] = log_posterior_1[accept]
        # update acceptance total numbers
        accp += np.sum(accept)

        if mcmc_step > burn_in_steps and (mcmc_step-burn_in_steps) % thin == 0:
            Theta_seq.append(Theta_proposal.copy())

    # Get MCMC samples after burn-in
    Theta_mcmc = tf.concat(Theta_seq, axis=0)
    accp_rate = accp / (N_proposal * (burn_in_steps + 1))
    print(f"Acceptance rate: {accp_rate:.4f}")

    return Theta_mcmc, accp_rate

# -----------------------------
# Run MCMC Refinement Multiple Times
# -----------------------------

Theta_mcmc, accp_rate = mcmc(N_proposal, burn_in_steps, n_samples, thin)
ps = Theta_mcmc.numpy()

# -----------------------------
# Save the posterior samples
# -----------------------------

posterior_samples_path = os.path.join(data_folder, "ps.npy")
np.save(posterior_samples_path, ps)

# Plot the posterior distribution
# 绘图
sns.set_style("whitegrid")

# 创建一个图形
fig, axs = plt.subplots(1, 5, figsize=(25, 4))

# 定义每个theta_i对应的x轴范围
x_limits = [
        [0.5, 1.5],  # theta_0
        [0.3, 1.7],  # theta_1
        [-2.0, 2.0],  # theta_2
        [-2.0, 2.0],  # theta_3
        [-0.3, 1.5],  # theta_4
    ]

for j, ax in enumerate(axs):
    ax.set_xlim(x_limits[j])
    ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

for j, ax in enumerate(axs):
    sns.kdeplot(
        ps[:, j],
        ax=ax,
        fill=False,
        label="Truth",
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

# 在每个子图上添加竖线表示真实参数的位置
for ax, true_p in zip(axs, true_ps):
    ax.axvline(true_p, color="r", linestyle="--", linewidth=1)

# 设置每个子图的标题
axs[0].set_title("theta1")
axs[1].set_title("theta2")
axs[2].set_title("theta3")
axs[3].set_title("theta4")
axs[4].set_title("theta5")

# 保存图片
plt.legend()
graph_path = os.path.join(current_dir, f"ps.png")
plt.savefig(graph_path)
plt.close()

# create a text file to record accp_rate
accp_rate_path = os.path.join(data_folder, "accp_rate.txt")
with open(accp_rate_path, "w") as f:
    f.write(f"Acceptance Rate: {accp_rate:.4f}\n")
