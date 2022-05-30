import numpy as np
from neural_network import NeuralNetwork 


def activation_function(value):
    return max(0, value)

def activation_function_der(value):
    return value > 0

def main():
    l_sizes = [5, 3, 4]
    predictions = np.array([0.45, 0.11, 0.3, 0.14])
    data = np.array([0.5, 0.3, 0.1, 0.1])
    nn = NeuralNetwork(l_sizes, activation_function, predictions, data, activation_function_der)
    

if __name__ == '__main__':
    main()