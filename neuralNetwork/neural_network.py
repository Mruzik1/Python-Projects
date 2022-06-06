import numpy as np
import random


class NeuralNetwork():
    # input data sh
    def __init__(self, act_func, act_func_der, input_data, predictions, layers_sizes):
        self.act_func = act_func
        self.act_func_der = act_func_der
        self.activation = np.vectorize(self.act_func)
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
        if layer:
            return self.weights[layer].dot(self.activation(self.nodes[layer])) + self.biases[layer]
        else:
            return self.weights[layer].dot(self.nodes[layer]) + self.biases[layer]


    # generates all nodes (just performs the forward propagation)
    def forward_pass(self):
        for l in range(len(self.weights)):
            self.nodes.append(self._forward_pass_step(l))


    # calculates the total error
    def total_error(self):
        
        return sum(1 / self.nodes[-1].size * (self.predictions-self.activation(self.nodes[-1]))**2)


    # computes the derivative of the total error with respect to any neuron
    def _error_neuron_der(self, n, layer):
        if layer == len(self.nodes)-1:
            return -2/len(self.nodes[layer])*(self.predictions[n]-self.act_func(self.nodes[layer][n]))
    
        result = list()

        for i in range(len(self.nodes[layer+1])):
            dzda = self.weights[layer][i][n]
            dadz = self.act_func_der(self.nodes[layer+1][i])
            result.append(self._error_neuron_der(i, layer+1) * dzda * dadz)

        return sum(result)


    # computes the derivative of the total error with respect to any weight
    def _error_weight_der(self, w, layer):
        dadz = self.act_func_der(self.nodes[layer+1][w[0]])
        dzdw = self.act_func(self.nodes[layer][w[1]]) if layer != 0 else self.nodes[layer][w[1]]
        dEda = self._error_neuron_der(w[0], layer+1)

        return dEda * dadz * dzdw


    # one step of back propagation
    def backprop(self, learning_speed):
        new_weights = list()

        for layer in range(len(self.nodes)-1, 0, -1):
            tmp = self.weights[layer-1].copy()

            for i in range(len(tmp)):
                for j in range(len(tmp[i])):
                    tmp[i][j] -= learning_speed * self._error_weight_der((i, j), layer-1)
            
            new_weights.append(tmp)

        new_weights.reverse()
        self.weights = new_weights
        self.nodes = [self.nodes[0]]