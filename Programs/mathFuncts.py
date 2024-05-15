import math


def activation(num):  # activation function
    return 1 / (1 + (math.e ** (0 - num)))


def label_to_vector(n):
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
    return sum([(expected_output[i][0] - actual_output[i][0]) ** 2 for i in range(10)])

