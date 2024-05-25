from random import random
from Programs.matrix import *


class Layer:  # takes input, multiplies it with weights, normalizes result, and outputs
    def __init__(self, column_count, row_count):
        self.weights = Matrix([])
        self.rows = row_count
        self.columns = column_count

        self.all_zeros(column_count, row_count)

    def __str__(self):
        return str(self.weights)

    def __getitem__(self, item):
        return self.weights[item]

    def all_zeros(self, x, y):
        self.weights = Matrix([[0 for i in range(x + 1)] for j in range(y)])  # x+1 to add a bias value

    def randomize_weights(self):
        self.weights = Matrix(
            [[(random() - .5) for i in range(self.weights.column_count())] for j in range(self.weights.row_count())]
        )

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
        self.layers = []
        self.layers.append(Layer(784, hidden_layer_1_size))
        self.layers.append(Layer(hidden_layer_1_size, hidden_layer_2_size))
        self.layers.append(Layer(hidden_layer_2_size, 10))

    def randomize_layers(self):
        for i in self.layers:
            i.randomize_weights()

    def raw_prediction(self, input_data):
        output = input_data
        for i in self.layers:
            output = i.feed_forward(output)
        return output

    def loss(self, labeled_data_batch):
        total = 0
        for i in range(labeled_data_batch.column_count()):
            total += difference_squared(
                digit_to_one_vector(
                    labeled_data_batch.column(i).get_label()
                ), self.raw_prediction(labeled_data_batch.column(i).get_data())
            )
        return total / labeled_data_batch.column_count()

    def save_weights(self):  # saves rows then columns, left to right, as if reading a book
        with open("LFS/weights_save.csv", "w") as file:
            for i in self.layers:
                file.write(weights_to_save_data(i.weights.items))
                file.write("\n")

    def load_weights(self):
        with open("LFS/weights_save.csv", "r") as file:
            raw_data = file.readlines()
            print(f"save data length: {len(raw_data)}")
            for i in range(len(self.layers)):
                x = self.layers[i].weights.column_count()
                y = self.layers[i].weights.row_count()
                self.layers[i].weights.items = Matrix(
                    [list(map(float, raw_data[i].split(",")))[x * j:x * (j + 1):] for j in range(y)]
                )

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
