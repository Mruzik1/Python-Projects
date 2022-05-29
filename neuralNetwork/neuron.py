class Neuron():
    # value - the neuron value
    # weights - neuron weights, numpy vector
    def __init__(self, value, weights):
        self.value = value
        self.weights = weights
    
    # returns weighted and activated values to the next neuron 
    def pass_value(self, activation_function):
        return [activation_function(e*self.value) for e in self.weights]

    # replace weight with new values
    def change_weights(self, new_weights):
        self.weights = new_weights
    
    # replace neuron value with new value
    def change_value(self, new_value):
        self.value = new_value