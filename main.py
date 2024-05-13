# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input (), 1st layer (16), 2nd layer (16), output (10)
# due to changes, this whole file will probably have to be redone

from LFS.Load_Data import *

print("\nStarted program, loading data & weights...\n")  # initialize program

layer1weights, layer2weights, layer3weights = rand_weights()  # randomly generates weight matrices
trainData = loadTrainingData()  # loads training data into a list of lists of brightness values
trainMatrix = [list(map(int, (i.strip().split(",")[1::] + ["1"]))) for i in trainData]  # Adds 1 for the bias value
displayLabels = [int(i[0]) for i in trainData]  # identifies the handwritten value in trainData
trainLabels = [label(j) for j in displayLabels]  # the correct output vector for each handwritten number
print(displayLabels[0])
print(type(displayLabels[0]))

print("Loading complete\n")


def convert(i):
    return i.index(max(i)) + 1


def run(inp):  # takes number values input, outputs list of predictions
    layer1activation = mat_mul(inp, layer1weights)
    layer2activation = mat_mul(layer1activation, layer2weights)
    final = mat_mul(layer2activation, layer3weights)
    return final


def loss():  # | || || |_
    res = 1
    outputs = [run(trainMatrix[i]) for i in range(len(trainMatrix))]
    total = list(map(lambda x: x, "fix this"))
    print(outputs[0])
    print(outputs[1])


print(run(trainMatrix[0]))
print(convert(run(trainMatrix[0])))

print("loss test start")
loss()
print("loss test end")
