import numpy as np
from scipy.stats import beta
import os
import pickle


def prior(batch_size):
    lambda_ = np.random.uniform(0, 1, size=batch_size)  # 感染率
    mu_ = np.random.uniform(0, 1, size=batch_size)  # 恢复率

    # lambda_ = np.random.lognormal(
    #     mean=np.log(0.4), sigma=0.5, size=batch_size
    # )  # 感染率
    # mu_ = np.random.lognormal(mean=np.log(1 / 8), sigma=0.2, size=batch_size)  # 恢复率

    theta = np.stack([lambda_, mu_], axis=1)
    return theta


def generate_observation(theta, T):
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

    for t in range(T):
        I_new = lambda_ * S * I

        # overflow protection
        I_new = np.where(I_new < S, I_new, S)

        S = S - I_new
        I = I + I_new - mu_ * I
        R = R + mu_ * I
        white_noise = np.random.normal(0, sigma, size=batch_size)  # 批量生成噪声
        I_new_obs = (1 + white_noise) * I_new
        I_new_obs = np.clip(I_new_obs, 0.0, 1.0)
        I_new_obs_list.append(I_new_obs)

    return np.array(I_new_obs_list).T


def generate_dataset(theta_candidate, T):
    X = generate_observation(theta_candidate, T)
    X = np.array(X)
    X = X.astype(np.float32)
    return X


batch_size = 12800
T = 100
n = T
d_x = 1

# data
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# generate training dataset
Theta = prior(batch_size)
X = generate_dataset(Theta, T)
X = X.reshape(batch_size, n, d_x)
XS = X.reshape(-1, n * d_x)
Theta = Theta.astype("float32")
X = X.astype("float32")
XS = XS.astype("float32")


# generate obs_xs
true_ps = np.array([[0.4, 0.15]])
obs_xs = generate_observation(true_ps, T)
np.save(os.path.join(data_folder, f"obs_xs.npy"), obs_xs)  # (n, d_x)

## training dataset for nn and score matching
x_train = np.concatenate((Theta, XS), axis=1)

# bayesflow
keys = [
    "prior_non_batchable_context",
    "prior_batchable_context",
    "prior_draws",
    "sim_non_batchable_context",
    "sim_batchable_context",
    "sim_data",
]

x_train_bf = dict.fromkeys(keys)
x_train_bf["prior_draws"] = Theta
x_train_bf["sim_data"] = X


np.save(os.path.join(data_folder, f"x_train.npy"), x_train)
file_path = os.path.join(data_folder, f"x_train_bf.pkl")
with open(file_path, "wb") as pickle_file:
    pickle.dump(x_train_bf, pickle_file)
