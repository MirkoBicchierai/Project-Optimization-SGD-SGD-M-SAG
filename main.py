from DataSet import DataSet
from Solver import Solver

if __name__ == '__main__':
    dataset = DataSet("DataSet/new_australian.csv", ",", 0)
    exe = Solver()

    exe.gd(dataset.data, dataset.label, 0.0001)

