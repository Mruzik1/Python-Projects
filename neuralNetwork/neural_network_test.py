import numpy as np
import matplotlib.pyplot as plt


# biases for the nn with layers' sizes: 2 2 2 plus biases
bias1 = 0.35
bias2 = 0.6

input_val = np.array([0.05, 0.1, bias1])
hidden = np.array([0.0, 0.0, bias2])
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


# makes one forward pass to the next layer, returns inactivated values
def forward_pass_step(layer, weights):
    return weights.dot(layer)


# calculate the total error: sum(1 / number_of_outputs * (predictions - output)^2)
def total_error(output, predictions):
    return sum(1 / output.size * (predictions-output)**2)


# TEST! counts the derivative of the total error with respect to any neuron
def error_neuron_der(weights, neurons, n, layer):
    if layer == len(neurons)-1:
        return -2/len(neurons[layer])*(predictions[n]-act_func(neurons[layer][n]))
    
    result = list()

    for i in range(len(neurons[layer])):
        dzda = weights[layer][i][n]
        dadz = act_func_der(neurons[layer+1][i])
        result.append(error_neuron_der(weights, neurons, i, layer+1) * dzda * dadz)

    return sum(result)


# TEST! counts the derivative of the total error with respect to any weight
def error_weight_der(weights, neurons, w, layer):
    dadz = act_func_der(neurons[layer+1][w[0]])
    dzdw = act_func(neurons[layer][w[1]]) if layer != 0 else neurons[layer][w[1]]

    return error_neuron_der(weights, neurons, w[0], layer+1) * dadz * dzdw


def backprop(weights, neurons, learning_speed):
    new_weights = list()

    for layer in range(len(neurons)-1, 0, -1):
        tmp = weights[layer-1].copy()

        for i in range(len(tmp)):
            for j in range(len(tmp[i])):
                tmp[i][j] -= learning_speed * error_weight_der(weights, neurons, [i, j], layer-1)
            
        new_weights.append(tmp)
    
    return new_weights


if __name__ == '__main__':
    activation = np.vectorize(act_func)
    errors = list()

    for _ in range(10000):
        
        # because we need same shapes and also the bias, let's append 0 at the end of the function's result
        hidden = np.append(forward_pass_step(input_val, w1), bias2)
        output_val = forward_pass_step(np.append(activation(hidden[:-1]), hidden[-1]), w2)

        # weights and neurons lists
        weights = [w1.T[:-1].T, w2.T[:-1].T]
        neurons = [input_val[:-1], hidden[:-1], output_val]

        errors.append(total_error(activation(output_val), predictions))
    
        weights = backprop(weights, neurons, 0.5)
        w1 = np.append(weights[0], np.array([[1], [1]]), 1)
        w2 = np.append(weights[1], np.array([[1], [1]]), 1)

        print(activation(output_val))
    
        plt.plot(errors)
        plt.pause(0.0005)
        
    plt.show()