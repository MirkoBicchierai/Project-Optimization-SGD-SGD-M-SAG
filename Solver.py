import numpy as np
from numpy import linalg as la


class Solver:
    def __init__(self, max_iteration, precision):
        self.max_iteration = max_iteration
        self.precision = precision

    def armijo(self, dataset, weights, direction):
        max_iter = 1000
        gamma = 0.5
        delta = 0.5
        alpha = 1

        old_loss = dataset.compute_log_loss(weights)
        grad = dataset.gradient(weights)
        k = 0

        new_weights = weights + alpha * direction
        new_loss = dataset.compute_log_loss(new_weights)
        while new_loss > old_loss + gamma * alpha * np.dot(grad.T, direction) and k < max_iter:
            alpha *= delta
            new_weights = weights + alpha * direction
            new_loss = dataset.compute_log_loss(new_weights)
            k = k + 1
        return alpha

    def gd(self, dataset):
        xk = np.zeros(dataset.data.shape[1])
        k = 0
        while la.norm(dataset.gradient(xk)) > self.precision and k < self.max_iteration:
            alpha = self.armijo(dataset, xk, -dataset.gradient(xk))
            xk = xk - alpha * dataset.gradient(xk)
            k = k + 1
            print(xk)
