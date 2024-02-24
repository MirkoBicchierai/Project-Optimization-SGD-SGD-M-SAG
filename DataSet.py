import numpy as np
from sklearn.datasets import load_svmlight_file


class DataSet:
    def __init__(self, percentage_split):
        self.percentage_split = percentage_split
        self.labels_train = []
        self.labels_test = []
        self.data_train = []
        self.data_test = []
        self.name = ""

    def load_data(self, path, name):
        self.name = name

        data = load_svmlight_file(path)
        tmp_x = data[0].toarray()
        tmp_y = data[1]

        indices = np.arange(len(tmp_x))
        np.random.shuffle(indices)
        tmp_x = tmp_x[indices, :]
        tmp_y = tmp_y[indices]

        split_index = int(0.8 * len(tmp_x))
        self.labels_train = tmp_y[:split_index]
        self.labels_test = tmp_y[split_index:]
        self.data_train = tmp_x[:split_index]
        self.data_test = tmp_x[split_index:]

    def fix(self, numer):
        self.labels_train[self.labels_train == numer] = -1
        self.labels_test[self.labels_test == numer] = -1
