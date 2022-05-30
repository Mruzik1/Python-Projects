import numpy as np
import random


class NeuralNetwork():
    def __init__(self, layers_sizes, act_func, predictions, data, act_func_der):
        self.act_func = act_func
        self.act_func_der = act_func_der
        self.predictions = predictions
        self.data = data
        self.weights = list()
        self.init_network(layers_sizes)

    # generates the neural network with random weights (and also with rand values, will be changed later)
    def init_network(self, layers_sizes):
        self.biases = np.array([np.array([0 for _ in range(e)]) for e in layers_sizes])
        for e1, e2 in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.weights.append(np.random.uniform(-0.5, 0.5, (e2, e1)))
        print(self.weights)
        print(self.biases)