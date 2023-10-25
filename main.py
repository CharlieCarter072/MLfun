# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input, 1st layer (16), 2nd layer (16), output (10)

from LFS.Load_Data import *
from Math.MathFuncts import *
from Math.Weights import *
import random

print("\nStarted program, loading data & weights...\n")

layer1weights, layer2weights, layer3weights = rand_weights()
trainData = loadTrainingData()
trainMatrix = [list(map(int, (i.strip().split(",")[1::] + ["1"]))) for i in trainData]
trainLabels = [i[0] for i in trainData]

print("Loading complete\n")



def run(inp):
    layer1Activation = mat_mulA(inp, layer1weights)
    layer2Activation = mat_mulA(layer1Activation, layer2weights)
    final = mat_mulA(layer2Activation, layer3weights)
    print(final)
    print(final.index(max(final)) + 1)


run(trainMatrix[0])
print(trainLabels[0])
