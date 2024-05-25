from LFS.load_data import *
from Programs.network import *


def main():
    print("\nStarted program, loading data & weights...\n")  # initialize program

    data_matrix = load_training_data_matrix()

    print("Loading complete\n")

    test_value = 6

    display_digit(data_matrix.column(test_value).get_data())

    testing_network = Network(16, 16)
    testing_network.load_weights()
    test_input = data_matrix.column(test_value).get_data()
    output = testing_network.raw_prediction(test_input)
    print(output)
    print(f"\nPrediction: {[output_to_digit(output)]}")
    print(f"Actual: {data_matrix.column(test_value).get_label()}\n")
    print(f"\nLabel test: {label_to_vector(data_matrix.column(test_value).get_label()[0])}")
    print(f"\nLoss test: {testing_network.loss(data_matrix)}")


main()



