import numpy as np


# biases for the nn with layers' sizes: 2 2 2 plus biases
input_val = np.array([0.05, 0.1, 0.35]).T
hidden = np.array([0, 0, 0.6]).T
output_val = np.array([0, 0]).T

# weights
w1 = np.array([[0.15, 0.2, 1], [0.25, 0.3, 1]])
w2 = np.array([[0.4, 0.45, 1], [0.5, 0.55, 1]])


# sigmoid
def activation_func(x):
    return 1 / (1 + np.exp(-x))


# makes one forward pass to the next layer
def forward_pass_step(layer, weights, activation_func):
    activation = np.vectorize(activation_func)

    return activation(weights.dot(layer))


print(forward_pass_step(input_val, w1, activation_func))