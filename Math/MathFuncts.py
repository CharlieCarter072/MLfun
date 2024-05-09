import math
from matrix import Matrix


def activation(num):
    return 1 / (1 + (math.e ** (0 - num)))


def mat_mul(a, b):
    finalMatrix = []
    for i in range(a.rows()):  # for rows of weights in a
        finalMatrix.append([dot_product(a[i], [row[column] for row in b]) for column in range(a.rows())])
    finalMatrix = Matrix(finalMatrix)
    return finalMatrix


def dot_product(a, b):  # dot product, scores similarity
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def label(n):
    easy = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],]
    return easy[n - 1]


def loss():
    pass
# loss function
#ugggh
#