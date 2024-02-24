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
    labels = ["SGD", "SGD-M", "SAG-B", "SAG-LS"] # "SAG" ,
    threshold = 0.5
    split = 0.8
    epochs = 75

    f = Function(threshold)
    dataset = DataSet(split)
    exe = Solver()

    print("-------------------------------------------------------------------")
    print("DataSet: a5a")

    dataset.load_data("DataSet/a5a", "a5a")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-2  # 1e-8
    beta = 0.3

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-1)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: phishing")

    dataset.load_data("DataSet/phishing", "phishing")
    dataset.fix(0)
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-1  # 1e-8
    beta = 0.20

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-1)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 10, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: ijcnn1")

    dataset.load_data("DataSet/ijcnn1", "ijcnn1")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-2  # 1e-8
    beta = 0.3

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 10, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: skin_nonskin")

    dataset.load_data("DataSet/skin_nonskin", "skin_nonskin")
    dataset.fix(2)
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-8  # 1e-8
    beta = 0.2

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e-4, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: german_numer_scale")

    dataset.load_data("DataSet/german_numer_scale", "german_numer_scale")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-2  # 1e-8
    beta = 0.15

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 0.5, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: skin_nonskin")

    dataset.load_data("DataSet/skin_nonskin", "skin_nonskin")
    dataset.fix(2)
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-8  # 1e-8
    beta = 0.2

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e-4, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: COD-RNA")

    dataset.load_data("DataSet/cod-rna", "cod-rna")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-7  # 1e-8
    beta = 0.25

    x_time_all = []
    y_loss_all = []
    x_step_all = []
    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e-4, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

    print("-------------------------------------------------------------------")
    print("DataSet: Australian")

    dataset.load_data("DataSet/australian_scale", "australian")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-2  # 1e-8
    beta = 0.25

    x_time_all = []
    y_loss_all = []
    x_step_all = []
    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

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
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1, batch_size)
    plot(dataset.name + "/sag_B_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_B_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-BATCH:", acc)

    print("SAG-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS(f, dataset, epochs)
    plot(dataset.name + "/sag_LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_LS_result_time.png", x_times, y_loss)
    # x_time_all.append(x_times)
    # y_loss_all.append(y_loss)
    # x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")
