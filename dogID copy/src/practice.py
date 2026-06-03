import random
import time

import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # randn(y, x) creates a y by x array of random numbers from normal dist
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    def sigmoid(z):
        return 1.0/(1.0+np/exp(-z))

    def feedforward(self, a):
        # given input a, compute output
        for b, w in zip(self.biases, self.weights):
            a = [sigmoid(np.dot(w, a) + b))]
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
