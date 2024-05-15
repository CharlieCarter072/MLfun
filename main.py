from LFS.Load_Data import *
from Programs.network import *


def main():
    print("\nStarted program, loading data & weights...\n")  # initialize program

    data_matrix = loadTrainingDataMatrix()  # should be testing data, but for home testing the testing data doesn't have a label, so it screws up


    print("Loading complete\n")

    test_value = 5

    display_digit(data_matrix.column(test_value).get_data())

    testing_network = Network(16, 16)  # clean
    test_input = data_matrix.column(test_value).get_data()
    output = testing_network.raw_prediction(test_input)
    print(output)
    print(f"\nPrediction: {[output_to_digit(output)]}")
    print(f"Actual: {data_matrix.column(test_value).get_label()}\n")
    print(f"\nLabel test: {label_to_vector(data_matrix.column(test_value).get_label()[0])}")
    print(f"\nLoss test: {testing_network.loss(data_matrix)}")


main()

