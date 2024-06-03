import math
from PIL import Image


class Matrix:
    def __init__(self, items=[]):
        self.items = items

    def __str__(self):
        return str(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def row_count(self):
        return len(list(self.items))

    def column_count(self):
        if self.row_count() == 0:
            return 0
        else:
            return len(list(self.items[0]))

    def row(self, row_index):
        return Matrix(self.items[row_index])

    def column(self, column_index):
        return Matrix([i[column_index] for i in self.items])

    def add_row(self, new_row):  # mostly useful for vectors, hopefully won't have to add an add_column method
        self.items.append(new_row)

    def get_label(self):
        return Matrix([self.items[0]])

    def get_data(self):
        return Matrix(self.items[1::])

    def edit_row(self, row_index, new_row):
        self.items[row_index] = new_row

    def edit_column(self, column_index, new_column):
        for i in range(self.row_count()):
            self.items[i][column_index] = new_column[i][0]

    def transpose(self):
        return Matrix([self.column(i).items for i in range(self.column_count())])

    def size(self):
        return f"{self.column_count()} x {self.row_count()}"


def mat_mul(a, b):
    final_matrix = Matrix(
        [[dot_product(a.row(i), b.column(j)) for j in range(b.column_count())] for i in range(a.row_count())]
    )
    return final_matrix


def vectorize(lst):
    return Matrix([[i] for i in lst])


def dot_product(a, b):  # dot product, scores similarity
    total = 0
    for i in range(a.row_count()):
        total += a[i] * b[i]
    return total


def all_zeros(x, y):
    return Matrix([[0 for i in range(x)] for j in range(y)])  # x+1 to add a bias value


def sigmoid(x):  # activation function
    return 1 / (1 + (math.e ** (0 - x)))


def sigmoid_prime(x):
    return Matrix([[sigmoid(j) * (1 - sigmoid(j)) for j in i] for i in x])


def digit_to_vector(n):
    labels = [[0] for i in range(10)]
    for i in range(len(labels)):
        if i == n:
            labels[i] = [1]
    return labels


def fix_brightness_values(x):
    return [int(x)/255]


def output_to_digit(output):
    prediction_value = 0
    for i in range(output.row_count()):
        if output[i][0] > output[prediction_value][0]:
            prediction_value = i
    return prediction_value


def difference_squared(expected_output, actual_output):
    return sum([((expected_output[i][0] - actual_output[i][0]) ** 2) / 10 for i in range(10)])


def difference_squared_prime(expected_output, actual_output):
    return Matrix([[2 * (actual_output[i][0] - expected_output[i][0]) / 10] for i in range(10)])


def image_to_data(file_name):
    image = Image.open(file_name)
    image = image.convert("RGB")
    print(f"img size: {image.size}")
    final_list = []
    for y in range(28):
        for x in range(28):
            r, g, b = image.getpixel((x, y))
            brightness = sum(image.getpixel((x, y))) / (3 * 255)
            final_list.append([brightness])
    return Matrix(final_list)
