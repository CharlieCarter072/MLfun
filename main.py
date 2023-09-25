# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input, 1st layer (16), 2nd layer (16), output (10)

from LFS.Load_Data import *
from Math.Math import *
from Math.Weights import *
import random


layer1weights, layer2weights, layer3weights = rand_weights()
trainData = loadTrainingData()
teast = trainData[0].split(",")
def run(inp):
    pass


print(trainData[0])
print(teast)
print(len(teast))
