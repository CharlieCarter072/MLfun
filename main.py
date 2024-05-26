from LFS.load_data import *
from Programs.network import *


def main():
    print("\nStarted program, loading data & weights...\n")  # initialize program

    data = load_training_data_matrix()

    print("Loading complete\n")

    testing_network = Network(16, 16)  # init network
    testing_network.randomize_layers()


    print("\n\nTraining network...\n")

    gens = 25

    for i in range(gens):
        testing_network.train_cycle(data, 64, 1)
        print(f"{100*(i+1)/gens}% complete\n")
    testing_network.save_weights()

    print("\nTraining complete\n")
    print("Testing trained network:\n")
    for n in range(10):
        test_input = data.column(n).get_data()  # test network prediction
        test_output = testing_network.predict(test_input)
        print(f"\nPrediction: {[output_to_digit(test_output)]}")
        print(f"Actual: {data.column(n).get_label()}\n")
        print(f"Output: {test_output}")


main()

