from random import random


class Layer:  # takes input, multiplies it with weights, normalizes result, and outputs
    def __init__(self, column_count, row_count):
        self.weights = []
        self.rows = row_count
        self.columns = column_count

        self.initialize_weights(column_count, row_count)

    def __str__(self):
        return str(self.weights)

    def __getitem__(self, item):
        return self.weights[item]

    def initialize_weights(self, x, y):
        self.weights = Matrix([[(random() - .5) for i in range(x + 1)] for j in range(y)])  # x+1 to add a bias value

    def edit_weight(self, x, y, new_value):
        self.weights[x][y] = new_value

    def feed_forward(self, data_in):  # input is a matrix
        data_in.add_row([1])
        unnormalized_output_data = mat_mul(self.weights, data_in)
        return [[activation(i[0])] for i in unnormalized_output_data]

    def backpropagation(self):
        pass  # ?????


class Network:
    pass
