from nnwv import *
import time

nn = NeuralNetwork([2, 8, 8, 1])
print(nn)

output = -1

while True:
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

        time.sleep(0.5)
    
    f = []
    for _ in input_:
        f.append(str(_))
    print(f"Result found for {' + '.join(f)}.")