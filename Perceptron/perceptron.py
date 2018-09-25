from enum import Enum


class Variant(Enum):
    DEFAULT = 1
    HALVING = 2


class Perceptron:

    def __init__(self, total_variables=0, variant=Variant.DEFAULT):
        # +1 of total variables in done to include bias term within the weight vector
        self.weights = [1] * (total_variables + 1)
        self.variant = variant

    def learn_weights(self):
        print(self.variant)


Perceptron()