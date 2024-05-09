from Math.matrix import *
import random


def rand_weights():
    # all layers for n-1 have one extra node with an activation of 1 for a bias
    layer1weights = [[(random.random() - .5) / 127.5 for i in range(785)] for j in range(16)]
    layer2weights = [[(random.random() - .5) for i in range(17)] for j in range(16)]
    layer3weights = [[(random.random() - .5) for i in range(17)] for j in range(10)]
    return [layer1weights, layer2weights, layer3weights]