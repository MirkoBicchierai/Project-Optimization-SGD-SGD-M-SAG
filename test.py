from DataSet import DataSet
from Function import Function
from Solver import Solver

if __name__ == '__main__':
    lamda = 0.8
    threshold = 0.5
    precision = 0.000001
    split = 0.8

    f = Function(lamda, threshold)
    dataset = DataSet(split)
    dataset.load_data("DataSet/cod-rna-mod.csv", ",", 0)
    exe = Solver(precision)

    exe.sgd(f, dataset, 50)
