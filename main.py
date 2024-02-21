from DataSet import DataSet
from Function import Function
from Solver import Solver
import matplotlib.pyplot as plt


def plot(file_name, x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.savefig("Plot/" + file_name)


if __name__ == '__main__':
    lamda = 0.8
    threshold = 0.5
    precision = 0.000001
    split = 0.8

    f = Function(lamda, threshold)
    dataset = DataSet(split)
    exe = Solver(precision)

    # COD-RNA DATASET
    dataset.load_data("DataSet/cod-rna.csv", ",", 0)
    lr = 1e-7  # 1e-8
    batch_size = 1
    epochs = 50
    beta = 0.25

    _, x_step, y_loss, x_times = exe.sgd(f, dataset, epochs, lr, batch_size)
    plot("cod-rna/sgd_result_" + str(lr) + "_" + str(batch_size) + ".png", x_step, y_loss)
    plot("cod-rna/sgd_result_" + str(lr) + "_" + str(batch_size) + "_time.png", x_times, y_loss)

    _, x_step, y_loss, x_times = exe.sgd_momentum(f, dataset, epochs, lr, batch_size, beta)
    plot("cod-rna/sgd_momentum_result_" + str(lr) + "_" + str(batch_size) + "_" + str(beta) + ".png", x_step, y_loss)
    plot("cod-rna/sgd_momentum_result_" + str(lr) + "_" + str(batch_size) + "_" + str(beta) + "_time.png", x_times, y_loss)
