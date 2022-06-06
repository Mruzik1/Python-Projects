import numpy as np
import random


class NeuralNetwork():
    # input data sh
    def __init__(self, act_func, act_func_der, input_data, predictions, layers_sizes):
        self.act_func = act_func
        self.act_func_der = act_func_der
        self.predictions = predictions
        self.nodes = [np.array(input_data)]
        self.weights = list()
        self.biases = list()                    # contains only bias weights
        self._init_network(layers_sizes)

    # generates the neural network with random weights
    def _init_network(self, layers_sizes):
        for l1, l2 in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.weights.append(np.random.random((l2, l1)))
            self.biases.append(random.random())

    # makes one forward pass to the next layer, returns inactivated values
    def _forward_pass_step(self, layer):
        return self.weights[layer] @ self.nodes[layer] + self.biases[layer]

    # generates all nodes (just performs the forward propagation)
    def forward_pass(self):
        activation = np.vectorize(self.act_func)

        for l in range(len(self.weights)):
            node = activation(self._forward_pass_step(l)) if l else self._forward_pass_step(l)
            self.nodes.append(node)

    # calculates the total error
    def total_error(self):
        activation = np.vectorize(self.act_func)
        
        return sum(1 / self.nodes[-1].size * (self.predictions-activation(self.nodes[-1]))**2)

    