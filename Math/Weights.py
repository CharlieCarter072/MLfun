from Math.Matrix import *
import random


def rand_weights():
    layer1weights = Matrix([[random.random() for i in range(784)] for j in range(16)])
    layer2weights = Matrix([[random.random() for i in range(16)] for j in range(16)])
    layer3weights = Matrix([[random.random() for i in range(16)] for j in range(10)])
    return [layer1weights, layer2weights, layer3weights]