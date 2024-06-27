import math
import random

class Neuron:
    def __init__(self, input_size):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)

    def forward(self, inputs):
        self.output = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return round(self.sigmoid(self.output), 4)

    def sigmoid(self, x):
        return 10 / (1 + math.exp(-x))

    def __str__(self) -> str:
        arr = 4
        w = []
        for wg in self.weights:
            w.append(str(round(wg, arr)))
        return f"\t\t[-] {', '.join(w)} ; {round(self.bias, arr)}"

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return self.outputs

    def __str__(self) -> str:
        sep = "\n"
        ne = []
        for neuron in self.neurons:
            ne.append(neuron.__str__())
        return f"\t[-] Layer (\n{sep.join(ne)}\n\t)"


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i+1], layer_sizes[i]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __str__(self) -> str:
        sep = "\n"
        ly = []
        for layer in self.layers:
            ly.append(layer.__str__())
        return f"[-] NeuralNetwork (\n{sep.join(ly)}\n)"
