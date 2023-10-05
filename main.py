# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input, 1st layer (16), 2nd layer (16), output (10)

from LFS.Load_Data import *
from Math.Math import *
from Math.Weights import *
import random

print("\nStarted program, loading data & weights...\n")

layer1weights, layer2weights, layer3weights = rand_weights()
trainData = loadTrainingData()
inputMatrix = [list(map(int, (i.strip().split(",")[1::] + ["1"]))) for i in trainData]
inputLabels = [i[0] for i in trainData]

print("Loading complete\n")
def run(inp):
    pass


print(len(inputMatrix[0]))
print(inputMatrix[0])
