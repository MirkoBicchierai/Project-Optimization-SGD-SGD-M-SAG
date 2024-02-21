import numpy as np
from numpy import linalg as la

from DataSet import sigmoid

threshold = 0.5


class Solver:
    def __init__(self, max_iteration, precision):
        self.max_iteration = max_iteration
        self.precision = precision

    def armijo(self, dataset, weights, direction):
        max_iter = 1000
        gamma = 0.3
        delta = 0.25
        alpha = 1
        k = 0

        old_loss = dataset.compute_log_loss(weights)
        grad = dataset.gradient2(weights)
        new_weights = weights + alpha * direction
        new_loss = dataset.compute_log_loss(new_weights)

        while new_loss > old_loss + gamma * alpha * np.dot(grad.T, direction) and k < max_iter:
            alpha *= delta
            new_weights = weights + alpha * direction
            new_loss = dataset.compute_log_loss(new_weights)
            k = k + 1
        return alpha

    def testing(self, dataset, weights):
        count = 0
        for i in range(dataset.data_test.shape[0]):
            prediction = self.predict(weights, dataset.data_test[i])
            if dataset.labels_test[i] == prediction:
                count = count + 1
        print("Accuracy: " + str(count / dataset.data_test.shape[0]) + "%")

    def predict(self, weights, data):
        if sigmoid(np.dot(weights, data)) >= threshold:
            return 1
        else:
            return -1

    def gd(self, dataset):
        xk = np.zeros(dataset.data_train.shape[1])
        k = 0
        while la.norm(dataset.gradient2(xk)) > self.precision and k < self.max_iteration:
            alpha = self.armijo(dataset, xk, -dataset.gradient2(xk))

            xk = xk - alpha * dataset.gradient2(xk)
            k = k + 1

        print(xk)
        self.testing(dataset, xk)
        if (k >= self.max_iteration):
            print("max iter cap")
        return xk

    def sgd(self, dataset, lr, epochs):
        xk = np.zeros(dataset.data_train.shape[1])
        k = 0

        while k < epochs and la.norm(dataset.gradient(dataset.data_train, dataset.labels_train,xk)) > self.precision:
            for i in range(dataset.data_train.shape[0]):
                xk = xk - lr * dataset.gradient2(xk)
            k = k + 1

        print(xk)
        self.testing(dataset, xk)
        return xk
