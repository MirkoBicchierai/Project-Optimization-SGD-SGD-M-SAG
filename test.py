from Function import Function
from Solver import Solver

if __name__ == '__main__':
    lamda = 0.8
    threshold = 0.5
    precision = 0.000001

    f = Function(lamda, threshold)
    exe = Solver(precision)