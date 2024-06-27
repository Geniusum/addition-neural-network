import math
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Neuron:
    def __init__(self, input_size):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.output = 0

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

class Visualizer:
    def __init__(self, neural_network):
        self.nn = neural_network
        self.fig = make_subplots(rows=1, cols=1)
        self.fig.update_layout(title='Neural Network Visualization', showlegend=False)
        
    def visualize(self, inputs):
        self.fig.data = []  # Clear previous data
        x, y = 0, 0
        
        for layer_idx, layer in enumerate(self.nn.layers):
            next_x = x + 1
            for neuron_idx, neuron in enumerate(layer.neurons):
                for weight_idx, weight in enumerate(neuron.weights):
                    self.fig.add_trace(go.Scatter(x=[x, next_x], y=[weight_idx, neuron_idx], 
                                                  mode='lines', line=dict(color='blue', width=abs(weight * 5))))
                    self.fig.add_trace(go.Scatter(x=[x, next_x], y=[weight_idx, neuron_idx], 
                                                  mode='markers', marker=dict(color='blue', size=5)))
                self.fig.add_trace(go.Scatter(x=[next_x], y=[neuron_idx], mode='markers+text',
                                              marker=dict(color='red', size=10),
                                              text=[round(neuron.output, 4)], textposition="bottom center"))
            x = next_x
        
        self.fig.show()

"""# Exemple d'utilisation
layer_sizes = [3, 5, 2]
nn = NeuralNetwork(layer_sizes)
visualizer = Visualizer(nn)

# Initialiser les entrées
inputs = [0.5, -0.2, 0.1]
outputs = nn.forward(inputs)

# Visualiser
visualizer.visualize(inputs)"""
