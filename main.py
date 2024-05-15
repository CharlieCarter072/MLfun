from LFS.Load_Data import *
from Programs.network import *


def main():
    print("\nStarted program, loading data & weights...\n")  # initialize program

    data_matrix = loadTrainingDataMatrix()  # should be testing data, but for home testing the testing data doesn't have a label, so it screws up


    print("Loading complete\n")

    test_value = 5

    display_digit(data_matrix.column(test_value).get_data())

    Testing_network = Network(16, 16)  # clean
    test_input = data_matrix.column(test_value).get_data()  # fix this to be in the right place (heck naw + vectorize)
    output = Testing_network.raw_prediction(test_input)
    print(output)
    print(f"\nPrediction: {[output_to_digit(output)]}")
    print(f"Actual: {data_matrix.column(test_value).get_label()}\n")  # heck naw * 2
    print(f"Test label: {data_matrix.column(test_value).get_label()}")
    print(f"Test data: {data_matrix.column(test_value).get_data()}")
    print(f"\nLabel test: {convert_to_label(data_matrix.column(test_value).get_label()[0])}")



main()

