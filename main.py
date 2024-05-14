# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input (), 1st layer (16), 2nd layer (16), output (10)
# due to changes, this whole file will probably have to be redone

from LFS.Load_Data import *
from Programs.network import *

print("\nStarted program, loading data & weights...\n")  # initialize program

data_matrix = loadTrainingDataMatrix()  # should be testing data, but for home testing the testing data doesn't have a label, so it screws up
data_matrix_no_labels = Matrix(data_matrix[1::])

print("Loading complete\n")

Testing_network = Network(16,16)
test_input = Matrix([[i] for i in data_matrix_no_labels.column(0)])
print(Testing_network.raw_prediction(test_input))
print(f"expected result: {data_matrix.items[0][0] * 255}")  # the * 255 is sketchy, but alright for now
