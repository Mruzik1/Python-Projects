import numpy as np
import random
from neuron_layer import NeuronsLayer 
from neuron import Neuron 


class NeuralNetwork():
    def __init__(self, layers_sizes, activation_function, softmax=True):
        self.layers_sizes = layers_sizes
        self.activation_function = activation_function
        self.softmax = softmax
        self.layers = list()
        self.generate_network()

    def generate_network(self):
        for i, e in enumerate(self.layers_sizes):
            neurons = list()
            for _ in range(e):
                weights = np.random.rand(1, self.layers_sizes[i+1]) if i < len(self.layers_sizes)-1 else [[]]
                value = random.randint(1, 10) if i == 0 else 0
                neurons.append(Neuron(value, weights[0]))
            self.layers.append(NeuronsLayer(neurons, self.activation_function))

    def print_network(self):
        for i, e in enumerate(self.layers):
            print(f'{i}) {[n.value for n in e.neurons]}')
    
    def process_step(self):
        for i in range(1, len(self.layers)):
            self.layers[i].change_neurons(self.layers[i-1].pass_values(), self.softmax)