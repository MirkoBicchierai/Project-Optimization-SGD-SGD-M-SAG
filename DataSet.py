import numpy as np


class DataSet:
    def __init__(self, path, delimiter, col_label):
        full_data = (np.genfromtxt(path, delimiter=delimiter, usemask=True))[1:]
        self.label = full_data[:, col_label]
        self.data = np.delete(full_data, col_label, axis=1)
