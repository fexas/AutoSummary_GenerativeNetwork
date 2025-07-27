import os
import numpy as np
import tensorflow as tf
from scipy.integrate import odeint
import random as rng
import pickle

N = 12800
n_points = 200  # n_points
n = n_points  # sample_size
d = 4  # dimension of parameter theta
d_x = 2  # dimenision of x
p = 9  # dimension of summary statistics
Q = 1  # number of draw from \exp{\frac{\Vert \theta_i - \theta_j \Vert^2}{w}} in first penalty
batch_size = 256
T_count = 15
x0 = 10  # initial number of prey
y0 = 5  # initial number of predator
p_lower = -5
p_upper = 2

true_ps = np.log(np.array([1, 0.01, 0.5, 0.01]))

# data
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# lotka_volterra_model
rng = np.random 
p_lower_list = np.array([p_lower]*d)
p_higher_list = np.array([p_upper]*d)


def _Prior(batch_size):
    theta = np.random.uniform(p_lower_list, p_higher_list, size=(batch_size, 4))
    return theta


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

    x, y, t = ecology_model(alpha, beta, gamma, delta, t_span=t_span, t_steps=t_steps, initial_state=initial_state)
    
    # add noise to the time series
    x += rng.normal(0, noise_scale, size=x.shape)
    y += rng.normal(0, noise_scale, size=y.shape)

    # concatenate the observed time series of x and y
    observed_X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

    return observed_X   

def simulate_lv_params(
    theta,
    n_points=500,
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


# gennerate obs_xs
obs_xs = lotka_volterra_forward(true_ps, n_points, T_count, x0, y0)
np.save(os.path.join(data_folder, f"obs_xs.npy"), obs_xs)  # （n, d_x）

# generate training datasets
Theta = _Prior(N)
X = simulate_lv_params(Theta, n_points, x0, y0, T_count, False)
XS = X.reshape(-1, n * d_x)
Theta = Theta.astype("float32")
X = X.astype("float32")
XS = XS.astype("float32")

## nn and score matching
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

print("obs_xs.shape: ", obs_xs.shape)
print("x_train.shape: ", x_train.shape)

np.save(os.path.join(data_folder, f"x_train.npy"), x_train)
file_path = os.path.join(data_folder, f"x_train_bf.pkl")
with open(file_path, "wb") as pickle_file:
    pickle.dump(x_train_bf, pickle_file)
