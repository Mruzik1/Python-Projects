import numpy as np
import random


class NeuralNetwork():
    # input data sh
    def __init__(self, act_func, act_func_der, layers_sizes):
        self.act_func = act_func
        self.act_func_der = act_func_der
        self.activation = np.vectorize(self.act_func)
        self.nodes = list()
        self.weights = list()
        self.biases = list()                    # contains only bias weights
        self._init_network(layers_sizes)


    # generates the neural network with random weights
    def _init_network(self, layers_sizes):
        for l1, l2 in zip(layers_sizes[:-1], layers_sizes[1:]):
            self.weights.append(np.random.random((l2, l1)))
            self.biases.append(np.random.random(l2))


    # makes one forward pass to the next layer, returns inactivated values
    def _forward_pass_step(self, layer):
        if layer:
            return self.weights[layer].dot(self.activation(self.nodes[layer])) + self.biases[layer]
        else:
            return self.weights[layer].dot(self.nodes[layer]) + self.biases[layer]


    # generates all nodes (just performs the forward propagation)
    def _forward_pass(self, input_data):
        self.nodes = [input_data]
        for l in range(len(self.weights)):
            self.nodes.append(self._forward_pass_step(l))


    # calculates the total error
    def _total_error(self, input_data, prediction):
        self._forward_pass(input_data)
        return sum(1 / self.nodes[-1].size * (prediction-self.activation(self.nodes[-1]))**2)


    # computes the derivative of the total error with respect to any neuron
    def _error_neuron_der(self, n, layer, prediction):
        if layer == len(self.nodes)-1:
            return -2/len(self.nodes[layer])*(prediction[n]-self.act_func(self.nodes[layer][n]))
    
        result = list()

        for i in range(len(self.nodes[layer+1])):
            dzda = self.weights[layer][i][n]
            dadz = self.act_func_der(self.nodes[layer+1][i])
            result.append(self._error_neuron_der(i, layer+1, prediction) * dzda * dadz)

        return sum(result)


    # computes the derivative of the total error with respect to any weight
    def _error_weight_der(self, weights, w_idx, layer, prediction):
        result = np.array([])
        dadz = self.act_func_der(self.nodes[layer+1][w_idx])
        dEda = self._error_neuron_der(w_idx, layer+1, prediction)

        for w in range(len(weights)):
            dzdw = self.act_func(self.nodes[layer][w]) if layer != 0 else self.nodes[layer][w]
            result = np.append(result, dEda * dadz * dzdw)

        return result

    # computes the derivative of the total error with respect to any bias
    def _error_bias_der(self, b, layer, prediction):
        return self.act_func_der(self.nodes[layer+1][b]) * self._error_neuron_der(b, layer+1, prediction)


    # one step of back propagation
    def backprop(self, learning_speed, input_data, prediction):
        new_weights = list()
        new_biases = list()
        self._forward_pass(input_data)

        for layer in range(len(self.nodes)-1, 0, -1):
            tmp_w = self.weights[layer-1].copy()
            tmp_b = self.biases[layer-1].copy()

            for i, e in enumerate(tmp_w):
                tmp_w[i] -= learning_speed * self._error_weight_der(e, i, layer-1, prediction)

            for i in range(len(tmp_b)):
                tmp_b[i] -= learning_speed * self._error_bias_der(i, layer-1, prediction)
            
            new_weights.append(tmp_w)
            new_biases.append(tmp_b)

        new_weights.reverse()
        new_biases.reverse()
        self.weights = new_weights
        self.biases = new_biases


    # repeats back propagation 'times' times, then returns weights, and prints the total error
    def train(self, times, learning_speed, predictions, training_data):
        print('The training has started, please wait...')

        for _ in range(times):
            for i, p in enumerate(predictions):
                for d in training_data[i]:
                    self.backprop(learning_speed, d, p)
        
        for i, p in enumerate(predictions):
            for d in training_data[i]:
                print(f'Total error for {d}: {self._total_error(d, p)}')

        return self.weights