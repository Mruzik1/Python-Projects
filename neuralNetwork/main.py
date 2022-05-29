from neural_network import NeuralNetwork 


def activation_function(value):
    return max(0, value)


def main():
    l_sizes = [4, 3, 5]
    nn = NeuralNetwork(l_sizes, activation_function)
    nn.print_network()
    print('-'*20)
    nn.process_step()
    nn.print_network()
    print('-'*20)
    nn.process_step()
    nn.print_network()
    print('-'*20)

if __name__ == '__main__':
    main()