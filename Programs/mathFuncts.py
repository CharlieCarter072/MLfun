import math  # use) still useful, file is modified so edits may have to be made
from UnitTests import Matrix


def activation(num):  # activation function
    return 1 / (1 + (math.e ** (0 - num)))


def mat_mul(a, b):
    final_matrix = Matrix([[dot_product(a.row(i), b.column(j)) for j in range(b.column_count())] for i in range(a.row_count())])
    return final_matrix


def dot_product(a, b):  # dot product, scores similarity
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def label(n):
    labels = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    return labels[n - 1]

