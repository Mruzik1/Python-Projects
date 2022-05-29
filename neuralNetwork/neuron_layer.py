import numpy as np


class NeuronsLayer():
    # neurons - array of neurons
    # activation_function - could be ReLU
    def __init__(self, neurons, activation_function):
        self.neurons = neurons
        self.activation_function = activation_function

    # returns new values array
    def pass_values(self):
        return sum(np.array([e.pass_value(self.activation_function) for e in self.neurons]))

    # new values for neurons
    def change_neurons(self, new_values):
        for i, e in enumerate(self.neurons):
            if i in range(len(new_values)): e.value += new_values[i]