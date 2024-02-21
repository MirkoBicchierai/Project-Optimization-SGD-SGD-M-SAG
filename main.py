import numpy as np
from DataSet import DataSet
from Solver import Solver

from scipy.optimize import minimize
from numpy import linalg as la

if __name__ == '__main__':
    seed_value = 27
    np.random.seed(seed_value)

    maxIterations = 1000
    precision = 0.0000001

    dataset = DataSet("DataSet/new_australian.csv", ",", 0)
    exe = Solver(maxIterations, precision)

    W1 = exe.gd(dataset)

    W2 = exe.sgd(dataset, 0.01, 10)

    w = minimize(dataset.compute_log_loss, np.zeros(dataset.data_train.shape[1]), jac=dataset.gradient)
    print("SCIPY", w.x)
    print("SGD:", la.norm(dataset.compute_log_loss(W2) - dataset.compute_log_loss(w.x)))
    print("GD:", la.norm(dataset.compute_log_loss(W1) - dataset.compute_log_loss(w.x)))
