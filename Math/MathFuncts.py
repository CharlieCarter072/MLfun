import math


def activation(num):
    return 1 / (1 + (math.e ** (0 - num)))


def mat_mulA(a, b):  # only for vector * screwed up matrix
    return [activation(dot_product(a, i)) for i in b]


def dot_product(a, b):  # dot product, scores similarity
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total