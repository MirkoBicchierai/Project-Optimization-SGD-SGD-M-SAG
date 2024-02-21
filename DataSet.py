import numpy as np


class DataSet:
    def __init__(self, percentage_split):
        self.percentage_split = percentage_split
        self.labels_train = []
        self.labels_test = []
        self.data_train = []
        self.data_test = []

    def load_data(self, path, delimiter, col_label):
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