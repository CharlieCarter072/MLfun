from random import random
from Programs.matrix import *


class Layer:  # takes input, multiplies it with weights, normalizes result, and outputs
    def __init__(self, column_count, row_count):
        self.weights = Matrix([])
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
        return Matrix([[sigmoid(i[0])] for i in unnormalized_output_data])

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

    def save_weights(self):  # saves rows then columns, left to right, as if reading a book
        with open("LFS/weights_save.csv", "w") as file:  #
            file.write(weights_to_save_data(self.layer_1.weights.items))
            file.write("\n")
            file.write(weights_to_save_data(self.layer_2.weights.items))
            file.write("\n")
            file.write(weights_to_save_data(self.layer_3.weights.items))

    def load_weights(self):
        with open("LFS/weights_save.csv", "r") as file:
            raw_data = file.readlines()
            self.layer_1.weights.items = Matrix(
                [list(map(float, raw_data[0].split(",")))[785 * i:785 * (i + 1):] for i in range(16)])
            self.layer_2.weights.items = Matrix(
                [list(map(float, raw_data[0].split(",")))[17 * i:17 * (i + 1):] for i in range(16)])
            self.layer_3.weights.items = Matrix(
                [list(map(float, raw_data[0].split(",")))[17 * i:17 * (i + 1):] for i in range(10)])

    def train_cycle(self, learning_rate, training_batch):
        pass
        # calculate gradient through partial derivatives
        # adjust weights by subtracting gradient * learning rate from existing weights
        # C is cost function, A is activation, W is weights, Z is weights * activation
        # dC/W = dZ/W * dA/Z * dC/A
        # dZ/W = previous layer activation
        # dA/Z = dSigma
        # dC/A = 2(activation - expected)


def weights_to_save_data(data):
    string_to_write = ""
    for i in data:
        for j in i:
            string_to_write += (str(j) + ",")
    return string_to_write[:-1:]


def cost(data_batch):
    pass  # 1/2 average of cost for each test
