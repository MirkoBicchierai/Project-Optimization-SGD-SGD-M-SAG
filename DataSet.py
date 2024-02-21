import numpy as np
from numpy import linalg as la

threshold = 0
lambda_reg = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DataSet:
    def __init__(self, path, delimiter, col_label):
        full_data = (np.genfromtxt(path, delimiter=delimiter, usemask=True))[1:]
        np.random.shuffle(full_data)
        split_index = int(0.8 * len(full_data))
        self.labels_train = full_data[:split_index, col_label]
        self.labels_test = full_data[split_index:, col_label]
        tmp = np.delete(full_data, col_label, axis=1)

        ones_column = np.ones((tmp.shape[0], 1))
        tmp_2 = np.hstack((tmp, ones_column))

        self.data_train = tmp_2[:split_index, :]
        self.data_test = tmp_2[split_index:, :]

    #
    # def gradient(self, X, y, weights):
    #     r = np.multiply(-y, sigmoid(np.multiply(-y, np.dot(X, weights))))
    #     return (np.matmul(X.T, r) * (1 / X.shape[0])) + lambda_reg * weights
    #
    # def compute_log_loss(self, weights):
    #     return ((1 / self.data_train.shape[0]) * np.sum(
    #         np.log(1 + np.exp(-self.labels_train * np.dot(self.data_train, weights)))) + lambda_reg / 2 * la.norm(
    #         weights) ** 2)
    #
    # def gradient2(self, weights):
    #     r = np.multiply(-self.labels_train, sigmoid(np.multiply(-self.labels_train, np.dot(self.data_train, weights))))
    #     return (np.matmul(self.data_train.T, r) * (1 / self.data_train.shape[0])) + lambda_reg * weights
