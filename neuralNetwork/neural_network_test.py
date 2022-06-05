import numpy as np


# biases for the nn with layers' sizes: 2 2 2 plus biases
input_val = np.array([0.05, 0.1, 0.35])
hidden = np.array([0.0, 0.0, 0.6])
output_val = np.array([0.0, 0.0])
predictions = np.array([0.01, 0.99])

# weights
w1 = np.array([[0.15, 0.2, 1], [0.25, 0.3, 1]])
w2 = np.array([[0.4, 0.45, 1], [0.5, 0.55, 1]])


# sigmoid
def act_func(x):
    return 1 / (1 + np.exp(-x))


# sigmoid derivative
def act_func_der(x):
    return np.exp(x) / (1 + np.exp(x))**2


# makes one forward pass to the next layer, return activated and inactivated values
def forward_pass_step(layer, weights):
    return weights.dot(layer)


# calculate the total error: sum(1 / number_of_outputs * (predictions - output)^2)
def total_error(output, predictions):
    return sum(1 / output.size * (predictions-output)**2)


# TESTING! counts the derivative of the total error with respect to any neuron
def error_neuron_der(weights, neurons, n, prev_n):
    if len(neurons) <= 1:
        dzda = weights[-1][n][prev_n]
        dadz = act_func_der(neurons[-1][n])
        dEda = -2/len(neurons[-1])*(predictions[n]-act_func(neurons[-1][n]))

        return dzda * dadz * dEda
    
    result = list()

    for i in range(len(neurons[1])):
        result.append(error_neuron_der(weights[1:], neurons[1:], i, n))

    return sum(result)


if __name__ == '__main__':
    activation = np.vectorize(act_func)

    # because we need same shapes and also the bias, let's append 0 at the end of the function's result
    hidden += np.append(forward_pass_step(input_val, w1), 0)
    output_val += forward_pass_step(np.append(activation(hidden[:-1]), hidden[-1]), w2)
    
    # seems to be right (error neuron derivative)
    print(error_neuron_der([w1.T[:-1].T, w2.T[:-1].T], [activation(hidden[:-1]), output_val], 0, 0))