from unittest import TestCase
from Programs.network import *
from LFS.load_data import *


class TestLayer(TestCase):
    test_layer = Layer(2, 3)

    for i in test_layer:
        #print(i)
        pass

    def test_initialize_weights(self):
        pass

    def test_edit_weight(self):
        pass

    def test_feed_forward(self):  # works (for now)
        test_input = Matrix([[1], [0.5], [-0.6]])
        #print(test_input)
        #print(self.test_layer.feed_forward(test_input))


class TestNetwork(TestCase):
    test_network = Network(16, 16)

    def test_raw_prediction(self):
        print(self.test_network.raw_prediction(Matrix([[1], [1], [0], [0], [1], [0], [1], [1], [1]])))

