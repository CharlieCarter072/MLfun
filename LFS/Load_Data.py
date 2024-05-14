from Programs.mathFuncts import *


def loadTrainingDataMatrix():  # row 0 is labels,
    with open("LFS/train.csv", "r") as train:
        full_unformatted_data = train.readlines()[:100:]  # the [:100:] is for fast loading purposes
        temp_matrix_full = Matrix([list(map(fix_brightness_values, (i.strip().split(",")))) for i in full_unformatted_data])  # messes up labels
        train_matrix_full = Matrix([temp_matrix_full.column(i) for i in range(785)])

        return train_matrix_full


def loadTestingDataMatrix():
    with open("LFS/test.csv", "r") as train:
        full_unformatted_data = train.readlines()[:100:]  # the [:100:] is for fast loading purposes

        temp_matrix_full = Matrix([list(map(fix_brightness_values, (i.strip().split(",")))) for i in full_unformatted_data])
        train_matrix_full = Matrix([temp_matrix_full.column(i) for i in range(785)])
        return train_matrix_full


def display_digit(data_in):
    for i in range(28):
        print([{0: " ", 1: "#"}[round(data_in[28*i + j][0])] for j in range(28)])


