import numpy as np
from numpy import linalg as la

threshold = 0
lambda_reg = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(data, label, weights):
    r = np.multiply(-label, sigmoid(np.multiply(-label, np.dot(data, weights))))
    return np.matmul(data.T, r) + 2 * lambda_reg * weights


def compute_log_loss(data, labels, weights):
    return (np.sum(np.log(1 + np.exp(-labels * np.dot(data, weights)))) + lambda_reg * la.norm(
        weights) ** 2) * (1 / data.shape[1])


class Solver:
    def __init__(self):
        pass

    def gd(self, data, label, lr):
        xk = np.zeros(14)
        while la.norm(gradient(data, label, xk)) > 0.00001:
            xk = xk - lr * gradient(data, label, xk)
            print(xk)
        print(xk)
