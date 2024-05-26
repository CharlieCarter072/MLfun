from Programs.math_functions import *


def load_training_data_matrix():  # row 0 is labels,
    with open("LFS/train.csv", "r") as train:
        full_unformatted_data = train.readlines()[:1000:]  # the [:100:] is for fast loading purposes
        temp_matrix_full = Matrix(
            [list(map(fix_brightness_values, (i.strip().split(",")))) for i in full_unformatted_data]
        )  # labels get screwed up as they are also divided by 255 when the brightness values get formatted
        temp_matrix_broken_labels = temp_matrix_full.transpose()
        temp_matrix_broken_labels.edit_row(0, [int(i[0] * 255) for i in temp_matrix_broken_labels[0]])

        return temp_matrix_broken_labels


def load_testing_data_matrix():
    with open("LFS/test.csv", "r") as train:
        full_unformatted_data = train.readlines()[:1000:]  # the [:100:] is for fast loading purposes
        temp_matrix_full = Matrix(
            [list(map(fix_brightness_values, (i.strip().split(",")))) for i in full_unformatted_data]
        )
        temp_matrix_broken_labels = temp_matrix_full.transpose()
        temp_matrix_broken_labels.edit_row(0, [int(i[0] * 255) for i in temp_matrix_broken_labels[0]])

        return temp_matrix_broken_labels


def display_digit(data_in):
    print("-"*(28*3 + 2))
    for i in range(28):
        temp_string = "|"
        for j in range(28):
            temp_string += {0: "   ", 1: "###"}[round(data_in[28*i + j][0])]
        temp_string += "|"
        print(temp_string)
    print("-" * (28 * 3 + 2))


