from random import random
from Programs.matrix import *


class Layer:  # takes input, multiplies it with weights, normalizes result, and outputs
    def __init__(self, column_count, row_count):
        self.weights = []
        self.rows = row_count
        self.columns = column_count

        self.randomize_weights(column_count, row_count)

    def __str__(self):
        return str(self.weights)

    def __getitem__(self, item):
        return self.weights[item]

    def randomize_weights(self, x, y):
        self.weights = Matrix([[(random() - .5) for i in range(x + 1)] for j in range(y)])  # x+1 to add a bias value

    def edit_weight(self, x, y, new_value):
        self.weights[x][y] = new_value

    def feed_forward(self, data_in):  # input is a matrix
        data_in.add_row([1])
        unnormalized_output_data = mat_mul(self.weights, data_in)
        return Matrix([[activation(i[0])] for i in unnormalized_output_data])

    def backpropagation(self):
        pass  # ?????

    def load_weights(self):
        pass


class Network:
    def __init__(self, hidden_layer_1_size, hidden_layer_2_size):
        self.layer_1 = Layer(784, hidden_layer_1_size)
        self.layer_2 = Layer(hidden_layer_1_size, hidden_layer_2_size)
        self.layer_3 = Layer(hidden_layer_2_size, 10)

    def raw_prediction(self, input_data):
        hidden_layer_1 = self.layer_1.feed_forward(input_data)
        hidden_layer_2 = self.layer_2.feed_forward(hidden_layer_1)
        return self.layer_3.feed_forward(hidden_layer_2)

    def loss(self, labeled_data_batch):
        total = 0
        for i in range(labeled_data_batch.column_count()):
            total += difference_squared(label_to_vector(labeled_data_batch.column(i).get_label()), self.raw_prediction(labeled_data_batch.column(i).get_data()))
        return total / labeled_data_batch.column_count()

    def train_cycle(self, learning_rate):
        pass

    def store_weights(self):
        pass

    def load_weights(self):
        pass

