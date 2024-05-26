from random import random, randint
from Programs.math_functions import *


class Layer:  # takes input, multiplies it with weights, normalizes result, and outputs
    def __init__(self, column_count, row_count):
        self.weights = Matrix([])
        self.delta_weights = all_zeros(column_count, row_count)
        self.rows = row_count
        self.columns = column_count
        self.input_A = Matrix([])
        self.input_B = Matrix([])
        self.passes = 0

        self.weights = all_zeros(column_count, row_count)

    def __str__(self):
        return str(self.weights)

    def __getitem__(self, item):
        return self.weights[item]

    def randomize_weights(self):
        self.weights = Matrix(
            [[(random() - .5) for i in range(self.weights.column_count())] for j in range(self.weights.row_count())]
        )

    def edit_weight(self, x, y, new_value):
        self.weights[x][y] = new_value

    def feed_forward(self, data_in):  # input is a matrix
        data_in.add_row([1])
        self.input_A = data_in
        self.input_B = mat_mul(self.weights, self.input_A)

        return Matrix([[sigmoid(i[0])] for i in self.input_B])

    def feed_backward(self, output_error):
        # Activation backfeed
        mid_error = all_zeros(output_error.column_count() - 1, output_error.row_count())

        for y in range(mid_error.row_count()):
            for x in range(mid_error.column_count()):
                mid_error[y][x] = sigmoid_prime(self.input_B)[y][x] * output_error[y][x]

        # Connected layer backfeed
        input_error = mat_mul(mid_error.transpose(), self.weights)

        weights_error = mat_mul(mid_error, self.input_A.transpose())  # if its not working, transpose self.input (switched?)

        for y in range(self.delta_weights.row_count()):
            for x in range(self.delta_weights.column_count()):
                self.delta_weights[y][x] += weights_error[y][x]

        self.passes += 1

        adjusted_error = Matrix([input_error.row(i)[:-1:] for i in range(input_error.row_count())]).transpose()

        return adjusted_error

    def step(self, learning_rate):
        for y in range(self.weights.row_count()):
            for x in range(self.weights.column_count()):
                self.weights[y][x] -= (learning_rate * self.delta_weights[y][x] / self.passes)
        self.delta_weights = all_zeros(self.weights.column_count() - 1, self.weights.row_count())
        self.passes = 0


class Network:
    def __init__(self, hidden_layer_1_size, hidden_layer_2_size):
        self.layers = []
        self.layers.append(Layer(784, hidden_layer_1_size))  # not exactly scalable...
        self.layers.append(Layer(hidden_layer_1_size, hidden_layer_2_size))
        self.layers.append(Layer(hidden_layer_2_size, 10))

    def randomize_layers(self):
        for i in self.layers:
            i.randomize_weights()

    def predict(self, input_data):
        output = input_data
        for i in self.layers:
            output = i.feed_forward(output)
        return output

    def save_weights(self):  # saves rows then columns, left to right, as if reading a book
        with open("LFS/weights_save.csv", "w") as file:
            for i in self.layers:
                file.write(weights_to_save_data(i.weights.items))
                file.write("\n")

    def load_weights(self):
        with open("LFS/weights_save.csv", "r") as file:
            raw_data = file.readlines()
            for i in range(len(self.layers)):
                x = self.layers[i].weights.column_count()
                y = self.layers[i].weights.row_count()
                self.layers[i].weights.items = Matrix(
                    [list(map(float, raw_data[i].split(",")))[x * j:x * (j + 1):] for j in range(y)]
                )

    def train_cycle(self, full_batch, batch_size, learning_rate):
        err = 0
        minibatch = Matrix(
            [full_batch.column(randint(0, full_batch.column_count() - 1)) for i in range(batch_size)]
        ).transpose()
        input_batch = minibatch.get_data()
        output_batch = minibatch.get_label()
        for i in range(batch_size):  # if reading from book, i = j as i is unused
            # output = input_batch[i]  # if it works, delete and replace output with predict(input_batch[i])
            # for layer in self.layers:
            #     output = layer.feed_forward(output)
            output = self.predict(input_batch.column(i))

            err += difference_squared(digit_to_vector(output_batch.column(i)), output)

            error = difference_squared_prime(digit_to_vector(output_batch.column(i)), output)
            for layer in reversed(self.layers):  # verbaitum, likely some errors
                error = layer.feed_backward(error)

        for layer in self.layers:
            layer.step(learning_rate)
        verbose = True
        if verbose:
            print(f"Error: {err / batch_size}")


def weights_to_save_data(data):
    string_to_write = ""
    for i in data:
        for j in i:
            string_to_write += (str(j) + ",")
    return string_to_write[:-1:]


def cost(data_batch):
    pass  # 1/2 average of cost for each test
