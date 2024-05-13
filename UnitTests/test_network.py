from unittest import TestCase


class TestLayer(TestCase):
    test_layer = Layer(2, 3)

    for i in test_layer:
        print(i)

    def test_initialize_weights(self):
        pass

    def test_edit_weight(self):
        pass

    def test_feed_forward(self):  # works (for now)
        test_input = Matrix([[1], [0.5], [-0.6]])
        print(test_input)
        print(self.test_layer.feed_forward(test_input))

