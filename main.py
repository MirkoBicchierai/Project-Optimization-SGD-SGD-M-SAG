import numpy as np
from DataSet import DataSet
from Function import Function
from Solver import Solver
import matplotlib.pyplot as plt


def plot(file_name, x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.grid(True)
    plt.savefig("Plot/" + file_name)


def plot_full(file_name, label, x, y, x_label):
    plt.subplots()
    x = np.array(x)
    y = np.array(y)
    for i in range(x.shape[0]):
        plt.plot(x[i], y[i], label=label[i])
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title('Training loss - ' + x_label)
    plt.legend()
    plt.grid(True)
    plt.savefig("Plot/" + file_name)


if __name__ == '__main__':
    lamda = 0.8
    threshold = 0.5
    precision = 0.000001
    split = 0.8
    epochs = 100

    f = Function(lamda, threshold)
    dataset = DataSet(split)
    exe = Solver(precision)

    # COD-RNA DATASET
    dataset.load_data("DataSet/cod-rna", "cod-rna")
    lr = 1e-7  # 1e-8
    beta = 0.25

    x_time_all = []
    y_loss_all = []
    x_step_all = []
    labels = ["SGD", "SGD-M", "SAG", "SAG-B", "SAG-LS"]

    print("DataSet: COD-RNA")

    print("SGD Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd(f, dataset, epochs, lr, 1)
    plot(dataset.name + "/sgd_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result_" + str(lr) + "_" + str(beta) + ".png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_" + str(lr) + "_" + str(beta) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-4)
    plot(dataset.name + "/sag_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e-4, batch_size)
    plot(dataset.name + "/sag_B_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_" + str(lr) + "_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("DataSet: Australian")
    dataset.load_data("DataSet/australian_scale", "australian")
    lr = 1e-8  # 1e-8
    beta = 0.2

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("SGD Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd(f, dataset, epochs, lr, 1)
    plot(dataset.name + "/sgd_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result_" + str(lr) + "_" + str(beta) + ".png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_" + str(lr) + "_" + str(beta) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-8)
    plot(dataset.name + "/sag_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e-6, batch_size)
    plot(dataset.name + "/sag_B_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_" + str(lr) + "_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result_" + str(lr) + ".png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_" + str(lr) + "_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)


    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")
