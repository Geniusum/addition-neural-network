import math
import random
import time
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

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

    def get_figure(self, inputs):
        fig = go.Figure()
        x, y = 0, 0

        for layer_idx, layer in enumerate(self.nn.layers):
            next_x = x + 1
            for neuron_idx, neuron in enumerate(layer.neurons):
                for weight_idx, weight in enumerate(neuron.weights):
                    fig.add_trace(go.Scatter(x=[x, next_x], y=[weight_idx, neuron_idx], 
                                             mode='lines', line=dict(color='blue', width=abs(weight * 5))))
                    fig.add_trace(go.Scatter(x=[x, next_x], y=[weight_idx, neuron_idx], 
                                             mode='markers', marker=dict(color='blue', size=5)))
                fig.add_trace(go.Scatter(x=[next_x], y=[neuron_idx], mode='markers+text',
                                         marker=dict(color='red', size=10),
                                         text=[round(neuron.output, 4)], textposition="bottom center"))
            x = next_x

        fig.update_layout(title='Neural Network Visualization', showlegend=False)
        return fig

# Initialiser le réseau de neurones et le visualiseur
nn = NeuralNetwork([2, 8, 8, 1])
visualizer = Visualizer(nn)
print(nn)

# Créer l'application Dash
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='graph-update',
        interval=500,  # Mise à jour toutes les 500 millisecondes
        n_intervals=0
    )
])

output = -1

@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update_graph_live(n):
    global output

    input_ = [random.randint(1, 4), random.randint(1, 4)]
    result = sum(input_)

    while round(output, 1) != result:
        output = nn.forward(input_)[0]
        print(output)

        if round(output, 1) != result:
            if output > result:
                for layers in nn.layers:
                    for neuron in layers.neurons:
                        for i, weight in enumerate(neuron.weights):
                            weight -= random.randint(100, 999) / 10000
                            neuron.weights[i] = weight
            elif output < result:
                for layers in nn.layers:
                    for neuron in layers.neurons:
                        for i, weight in enumerate(neuron.weights):
                            weight += random.randint(100, 999) / 10000
                            neuron.weights[i] = weight

    return visualizer.get_figure(input_)

if __name__ == '__main__':
    app.run_server(debug=True)
