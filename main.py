# input vector, multiply with weights -> activation for next layer (don't forget bias)
# input, 1st layer (16), 2nd layer (16), output (10)

import time
import random
import math


def loadTrainingData():
    with open("train.csv", "r") as train:
        return train.readlines()


def loadTestingData():
    with open("test.csv", "r") as train:
        return train.readlines()


def activation(num):
    return 1 / (1 + (math.e ** (0 - num)))


class Matrix(list):  # A = Matrix([[1, 2],[3, 4]])     A @ B
    def __matmul__(self, B):
        A = self
        return Matrix([[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0])) ] for i in range(len(A))])


layer1Weights = Matrix([[random.random() for i in range(784)] for j in range(16)])
layer2Weights = Matrix([[random.random() for i in range(16)] for j in range(16)])
layer3Weights = Matrix([[random.random() for i in range(16)] for j in range(10)])

trainData = loadTrainingData()
teast = trainData[0].split(",")
def run(inp):
    pass


print(trainData[0])
print(teast)
print(len(teast))
