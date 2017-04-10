#!/usr/bin/env python3
import neural_network as nn
import random


def draw_line(dots):
    line = ""
    for dot in dots:
        if dot == 0:
            symbol = " "
        else:
            symbol = "_"
        line += symbol
    return line


def generate_solid_line(length):
    line = [0] * random.choice([0] * 5 + [1, 2, 3, 4, 5, 6, 7, 8]) * (1 + length // 40)
    line_end = [0] * random.choice([0] * 5 + [1, 2, 3, 4, 5, 6, 7, 8]) * (1 + length // 40)
    while len(line) + len(line_end) < length:
        line += [random.choice([1] * 15 + [0])] * random.randint(0, length // 5)
    line = line[:length - len(line_end)] + line_end
    return line


def generate_dotted_line(length):
    line = []
    dot = random.choice([0, 1])
    while len(line) < length:
        line += [dot] * random.choice([1] * 200 + [2] * 10 + [3] * 2 + [4])
        dot = abs(dot - 1)
    return line[:length]


random.seed(8)
learning_set_1 = [([0, 1, 2, 3, 4], 5), ([-10, -5, 0, 5, 10], 15), ([70, 80, 90, 100, 110], 120)]
learning_set = [((1, 2, 3, 4, 5), (1, 0)), ((5, 4, 3, 2, 1), (0, 1)),
                ((-50, -20, 30, 40, 50), (1, 0)), ((135, 40, 60, 20, 1), (0, 1))]
test_data = [100, 200, 300, 400, 500]

n = nn.Network(input_count=10, layers=[nn.Layer(nn.Neuron(nn.logistic), 10),
                                       nn.Layer(nn.Neuron(nn.logistic), 2)])
n.fully_link()
n.initialize_random()

for _ in range(20000):
    n.learn(generate_solid_line(10), (1, 0))
    n.learn(generate_dotted_line(10), (0, 1))

test_count = 1000
solid_output = [sum(map(lambda x: x[1] if x[0] == 0 else -x[1],
                        enumerate(n <= generate_solid_line(10)))) > 0
                for _ in range(test_count)]
dotted_output = [sum(map(lambda x: x[1] if x[0] == 0 else -x[1],
                         enumerate(n <= generate_dotted_line(10)))) < 0
                 for _ in range(test_count)]
solid_percentage = sum(solid_output) / test_count
dotted_percentage = sum(dotted_output) / test_count
print(solid_percentage, "\t", dotted_percentage)
