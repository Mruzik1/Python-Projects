from neural_network import NeuralNetwork
import math


# sigmoid
def act_func(x):
    return 1 / (1 + math.exp(-x))


# sigmoid derivative
def act_func_der(x):
    return math.exp(x) / (1 + math.exp(x))**2


if __name__ == '__main__':
    input_data = [0.05, 0.1]
    predictions = [0.01, 0.99]
    sizes = [2, 8, 6, 2]
    nn = NeuralNetwork(act_func, act_func_der, input_data, predictions, sizes)

    nn.forward_pass()
    print(nn.total_error())