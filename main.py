from LFS.Load_Data import *
from Programs.network import *


def main():
    print("\nStarted program, loading data & weights...\n")  # initialize program

    data_matrix = loadTrainingDataMatrix()  # should be testing data, but for home testing the testing data doesn't have a label, so it screws up
    data_matrix_values = Matrix(data_matrix[1::])  # heck naw (should never see matrix function outside matrix class
    data_matrix_labels = data_matrix.row(0)

    print("Loading complete\n")

    test_value = 3

    Testing_network = Network(16, 16)  # clean
    test_input = vectorize(data_matrix_values.column(test_value))  # fix this to be in the right place (heck naw + vectorize)
    print(Testing_network.raw_prediction(test_input))  # (heck naw pt. 2)
    print(f"expected result: {data_matrix.items[0][test_value] * 255}")  # heck naw * 2
    display_digit(vectorize(data_matrix_values.column(test_value)))


main()

