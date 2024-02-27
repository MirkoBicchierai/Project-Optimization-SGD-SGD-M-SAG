import numpy as np
from DataSet import DataSet
from Function import Function
from Solver import Solver
import matplotlib.pyplot as plt
import scipy.optimize as sc

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

def skin_nonskin(f,dataset,epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 0.0000001)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def german_numer_scale(f,dataset,epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 4*1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 0.01)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def phishing(f, dataset, epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 4 * 1e1)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 7 * 0.01)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 1e1, batch_size)
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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def ijcnn1(f, dataset, epochs):
    print("-------------------------------------------------------------------")
    print("DataSet: ijcnn1")

    dataset.load_data("DataSet/ijcnn1", "ijcnn1")
    f.set_lamda(1 / dataset.data_train.shape[0])
    lr = 1e-2  # 1e-8
    beta = 0.35

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    print("Samples: " + str(dataset.data_train.shape[0]) + "  features: " + str(dataset.data_train.shape[1]))
    print("-------------------------------------------------------------------")

    print("SGD Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd(f, dataset, epochs, lr, 1)
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e1)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 4 * 0.01)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

    print("SAG-BATCH Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_B(f, dataset, epochs, 2, batch_size)
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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def a5a(f, dataset, epochs):
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
    # w = np.zeros(dataset.data_train.shape[1], dtype="float128")
    # optimal_point = sc.minimize(f.loss_function, w, method='L-BFGS-B').x
    # print(optimal_point)
    print("SGD Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd(f, dataset, epochs, lr, 1)
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 0.1)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def skin_nonskin(f, dataset, epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-2)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 3 * 0.0000001)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def cod_rna(f, dataset, epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-4)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 0.000001)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

def australian(f, dataset, epochs):
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
    plot(dataset.name + "/sgd_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD:", acc)

    print("SGD-M Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr, 1, beta)
    plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
    plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SGD Momentum:", acc)

    print("SAG Algorithms")
    batch_size = int(dataset.data_train.shape[0] / 4)
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, 1e-3)
    plot(dataset.name + "/sag_result.png", x_step, y_loss)
    plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG:", acc)

    print("SAGV2 Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs, 0.01)
    plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2:", acc)

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
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAG-LS:", acc)

    print("SAGV2-LS Algorithms")
    _, x_step, y_loss, x_times, acc = exe.sag_algorithm_LS_2(f, dataset, epochs)
    plot(dataset.name + "/sagV2-LS_result.png", x_step, y_loss)
    plot(dataset.name + "/sagV2-LS_result_time.png", x_times, y_loss)
    x_time_all.append(x_times)
    y_loss_all.append(y_loss)
    x_step_all.append(x_step)
    print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")

if __name__ == '__main__':
    labels = ["SGD", "SGD-M", "SAG", "SAGV2", "SAG-B", "SAG-LS", "SAGV2-LS"]
    threshold = 0.5
    split = 0.8
    epochs = 50

    f = Function(threshold)
    dataset = DataSet(split)
    exe = Solver()

    australian(f, dataset, epochs) # ok
    # skin_nonskin(f,dataset,epochs) #OK
    # german_numer_scale(f, dataset, epochs) #  ok
    # phishing(f, dataset, epochs) # sus
    # ijcnn1(f, dataset, epochs) # ok
    # a5a(f, dataset, epochs) # ok
    # cod_rna(f, dataset, epochs) # sus









