import numpy as np
from numpy import linalg as la

threshold = 0
lambda_reg = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DataSet:
    def __init__(self, path, delimiter, col_label):
        full_data = (np.genfromtxt(path, delimiter=delimiter, usemask=True))[1:]
        self.labels = full_data[:, col_label]
        self.data = np.delete(full_data, col_label, axis=1)

    def gradient(self, weights):
        r = np.multiply(-self.labels, sigmoid(np.multiply(-self.labels, np.dot(self.data, weights))))
        return np.matmul(self.data.T, r) + 2 * lambda_reg * weights

    def compute_log_loss(self, weights):
        return (np.sum(np.log(1 + np.exp(-self.labels * np.dot(self.data, weights)))) + lambda_reg * la.norm(
            weights) ** 2) * (1 / self.data.shape[1])
