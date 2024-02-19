from DataSet import DataSet
from Solver import Solver

if __name__ == '__main__':
    maxIterations = 1000
    precision = 0.00001

    dataset = DataSet("DataSet/new_australian.csv", ",", 0)
    exe = Solver(maxIterations, precision)

    exe.gd(dataset)
