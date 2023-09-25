def loadTrainingData():
    with open("LFS/train.csv", "r") as train:
        return train.readlines()


def loadTestingData():
    with open("LFS/test.csv", "r") as train:
        return train.readlines()