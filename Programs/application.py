from LFS.load_data import *
from Programs.network import *


def application():
    print("\nStarted program, loading data...")
    train_data = load_training_data_matrix(True)
    #test_data = load_testing_data_matrix(True)  # not fully working
    print("Loading complete")

    network = Network()
    network.add_layer(784, 16)
    network.add_layer(16, 16)
    network.add_layer(16, 10)

    active = True
    while active:
        choice_a = input(
            "\n(1) Exit\n"
            "(2) Network manager\n"
            "(3) Test network\n"
            "(4) Train network\n"
            ">>> "
        )
        match choice_a:
            case "1":
                active = False
            case "2":
                while active:
                    choice_b = input(
                        f"\nNetwork layers: {[i.size() for i in network.layers]}\n"
                        "(1) Back\n"
                        "(2) Add layer\n"
                        "(3) Remove layer\n"
                        "(4) Randomize weights\n"
                        "(5) Set all weights to 0\n"
                        "(6) Load trained weights\n"
                        "(7) Load saved weights\n"
                        "(8) Save weights\n"
                        ">>> "
                    )
                    match choice_b:
                        case "1":
                            active = False
                        case "2":
                            input_size = int(input("\nLayer input size\n>>> "))
                            output_size = int(input("\nLayer output size\n>>> "))
                            network.add_layer(input_size, output_size)
                            print("\nLayer successfully added")
                        case "3":
                            network.remove_layer()
                            print("\nLayer successfully removed")
                        case "4":
                            network.randomize_weights()
                            print("\nWeights successfully randomized")
                        case "5":
                            network.all_zeros()
                            print("\nWeights successfully set to 0")
                        case "6":
                            network.load_weights("LFS/weights_save_trained.csv")
                            print("\nWeights successfully loaded")
                        case "7":
                            network.load_weights("LFS/weights_save.csv")
                            print("\nWeights successfully loaded")
                        case "8":
                            network.save_weights("LFS/weights_save.csv")
                            print("\nWeights successfully saved")
                        case _:
                            print("\nInvalid input")
                active = True
            case "3":
                display_details = {"y": True, "n": False}[
                    input("\nWould you like to display extra info? (y/n)\n>>> ").lower()
                ]
                while active:
                    choice_b = input(
                        "\n(1) Back\n"
                        "(2) Test with random data sample\n"
                        "(3) Test with custom image\n"
                        ">>> "
                    )
                    match choice_b:
                        case "1":
                            active = False
                        case "2":
                            test_sample = train_data.column(randint(0, train_data.column_count() - 1))
                            test_output = network.predict(test_sample.get_data())
                            print("")
                            if display_details:
                                display_digit(test_sample.get_data())
                                print(test_output)
                            print(f"\nNetwork prediction: {output_to_digit(test_output)}")
                            print(f"Expected result: {test_sample.get_label()[0]}")
                        case "3":
                            pass  # good luck with this one
                        case _:
                            print("\nInvalid input")
                active = True
            case "4":
                print("\nLoading training dataset (this may take a while)...")
                train_data = load_training_data_matrix()
                print("Loading complete")
                network.train_cycle(train_data)
            case _:
                print("\nInvalid input")

