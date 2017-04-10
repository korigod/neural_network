#!/usr/bin/env python3
import math
import random


LEARNING_RATE = 0.02


def debug_print(*args):
    return print(*args)


def logistic(x, derivative=False):
    'When derivative=True, x should be function output, not argument.'
    if not derivative:
        # debug_print(x)
        return 1 / (1 + math.exp(-x)) if x > -100 else 0
    else:
        return x * (1 - x)


def ReLU(x, derivative=False):
    if not derivative:
        return max(0, x)
    else:
        return 1 if x > 0 else 0


class BaseNeuron:

    def __init__(self):
        self.input_neurons = []
        self.output_neurons = []
        self.input_d = {}
        self.last_input_d = {}
        self.weight_d = {}
        self.bias = None
        self.activ_func = None
        self.delta = None
        self.backpropagated_d = {}

        # Used for output neurons only
        self.output = None
        self.expected_output_d = {}

    def connect_output_to(self, next_neuron):
        self.output_neurons.append(next_neuron)
        next_neuron.input_neurons.append(self)

    def initialize_random(self):
        for neuron in self.input_neurons:
            self.weight_d[neuron] = 0.1 * random.gauss(0, 1) / math.sqrt(len(self.input_neurons))
        self.bias = 0

    def excite(self, sending_neuron, value):
        if sending_neuron in self.input_neurons:
            self.input_d[sending_neuron] = value
        # debug_print(self, " feels excited! ", self.input_d)
        if len([1 for v in self.input_d.values() if v is not None]) == len(self.input_neurons):
            # debug_print(self, " I'm all right!")
            self.live()
            self.relax()

    def live(self):
        raise NotImplementedError

    def relax(self):
        self.last_input_d = self.input_d.copy()
        self.input_d = dict.fromkeys(self.input_neurons, None)

    def excite_back(self, sending_neuron, value):
        if isinstance(sending_neuron, OutputNeuron):
            expected_output = value
            self.delta = (expected_output - self.output) * self.activ_func(self.output, derivative=True)
            self.backpropagate()
        elif isinstance(sending_neuron, Neuron):
            self.backpropagated_d[sending_neuron] = value
            if len([1 for v in self.backpropagated_d.values()
                    if v is not None]) == len(self.output_neurons):
                self.delta = sum(map(lambda val: val['weight'] * val['delta'],
                                     self.backpropagated_d.values())) * self.activ_func(
                                         self.output, derivative=True)
                self.backpropagate()

    def backpropagate(self):
        for neuron in self.input_neurons:
            neuron.excite_back(self, {'delta': self.delta, 'weight': self.weight_d[neuron]})
            self.weight_d[neuron] += LEARNING_RATE * self.delta * neuron.output

    def clone(self):
        cloned = self.__class__()
        cloned.input_neurons = self.input_neurons[:]
        cloned.output_neurons = self.output_neurons[:]
        cloned.input_d = self.input_d.copy()
        cloned.last_input_d = self.last_input_d.copy()
        cloned.weight_d = self.weight_d.copy()
        cloned.bias = self.bias
        cloned.activ_func = self.activ_func
        cloned.output = self.output
        cloned.expected_output_d = self.expected_output_d
        cloned.delta = self.delta
        return cloned


class Neuron(BaseNeuron):

    def __init__(self, activation_fuction=logistic):
        super().__init__()
        self.activ_func = activation_fuction

    def live(self):
        raw_output = self.bias
        for sending_neuron, input_val in self.input_d.items():
            raw_output += input_val * self.weight_d[sending_neuron]
        # debug_print(raw_output)
        self.output = self.activ_func(raw_output)
        for neuron in self.output_neurons:
            neuron.excite(self, self.output)


class InputNeuron(BaseNeuron):
    def __init__(self):
        super().__init__()
        self.input_neurons = ["input"]
        self.input_d = {"input": None}

    def __le__(self, value):
        self.excite("input", value)

    def excite_back(self, sending_neuron, value):
        # debug_print("Input excited back!")
        pass

    def live(self):
        self.output = self.input_d["input"]
        for neuron in self.output_neurons:
            neuron.excite(self, self.output)


class OutputNeuron(BaseNeuron):

    def __init__(self):
        super().__init__()
        self.output_d = None

    def live(self):
        self.output_d = self.input_d.copy()

    def backpropagate(self):
        for neuron in self.input_neurons:
            neuron.excite_back(self, self.expected_output_d[neuron])


class BaseLayer:

    def __init__(self, neurons):
        self.neurons = neurons

    def initialize_random(self):
        for neuron in self.neurons:
            neuron.initialize_random()

    def fully_link_to(self, next_layer):
        for my_neuron in self.neurons:
            for next_layer_neuron in next_layer.neurons:
                my_neuron.connect_output_to(next_layer_neuron)


class Layer(BaseLayer):

    def __init__(self, neuron, count):
        super().__init__([neuron.clone() for _ in range(count)])


class Network:

    def __init__(self, input_count, layers):
        self.input_count = input_count
        self.layers = [Layer(InputNeuron(), input_count)]
        self.layers.extend(layers)
        # self.layers.append(Layer(OutputNeuron(), len(layers[-1].neurons)))
        self.layers.append(Layer(OutputNeuron(), 1))
        self.output = None

    def __le__(self, inputs):
        return self.excite(inputs)

    def fully_link(self):
        for some_layer, next_layer in zip(self.layers, self.layers[1:]):
            some_layer.fully_link_to(next_layer)

    def initialize_random(self):
        for layer in self.layers:
            layer.initialize_random()

    def excite(self, inputs):
        for index, neuron in enumerate(self.layers[0].neurons):
            neuron <= inputs[index]
        # return [output_neuron.output for output_neuron in self.layers[-1].neurons]
        output_neuron = self.layers[-1].neurons[0]
        return [output_neuron.output_d[neuron] for neuron in output_neuron.input_neurons]

    def learn(self, inputs, expected_outputs):
        out = self <= inputs
        debug_print(out, "\t", expected_outputs)
        self.layers[-1].neurons[0].expected_output_d = {neuron: expected_outputs[index] for index, neuron
                                                        in enumerate(self.layers[-2].neurons)}
        self.layers[-1].neurons[0].backpropagate()
