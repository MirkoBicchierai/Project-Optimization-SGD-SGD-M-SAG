import numpy as np
from DataSet import DataSet
from Function import Function
from Solver import Solver
import matplotlib.pyplot as plt
import scipy.optimize as sc

list_w_LBFGSB = []


def print_callback(wk):
    global list_w_LBFGSB
    list_w_LBFGSB.append(wk)


def fill_to_n_with_last_element(lst, N):
    if len(lst) >= N:
        return lst[:N]
    else:
        last_element = lst[-1] if lst else 0
        return lst + [last_element] * (N - len(lst))


def plot(file_name, x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    plt.grid(True)
    plt.savefig("Plot/" + file_name)


def plot_full(file_name, label, x, y, x_label):
    plt.subplots()
    x = np.array(x)
    y = np.array(y)
    if x_label == "Time":
        for i in range(x.shape[0]):
            plt.plot(x[i], y[i], label=label[i + 1])
    else:
        for i in range(x.shape[0]):
            plt.plot(x[i], y[i], label=label[i])
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title('Training loss - ' + x_label)
    plt.legend()
    plt.grid(True)
    plt.savefig("Plot/" + file_name)


def phishing(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: phishing")

    dataset.load_data("DataSet/phishing", "phishing")
    dataset.fix(0)
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.20
    lr_sag = 1e-4
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def ijcnn1(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: ijcnn1")

    dataset.load_data("DataSet/ijcnn1", "ijcnn1")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.35
    lr_sag = 1e-4
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def skin_nonskin(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: skin_nonskin")

    dataset.load_data("DataSet/skin_nonskin", "skin_nonskin")
    dataset.fix(2)
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-8
    beta_sgd = 0.20
    lr_sag = 1e-9
    lr_sagv2 = 1e-9

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def cod_rna(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: COD-RNA")

    dataset.load_data("DataSet/cod-rna", "cod-rna")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-7
    beta_sgd = 0.25
    lr_sag = 1e-2
    lr_sagv2 = 1e-3

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def german_numer_scale(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: german_numer_scale")

    dataset.load_data("DataSet/german_numer_scale", "german_numer_scale")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 5 * 1e-4
    beta_sgd = 0.15
    lr_sag = 1e-3
    lr_sagv2 = 1e-4

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def australian(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: Australian")

    dataset.load_data("DataSet/australian_scale", "australian")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 2*1e-4
    beta_sgd = 0.4
    lr_sag = 3*1e-4
    lr_sagv2 = 1e-4

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def a5a(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: a5a")

    dataset.load_data("DataSet/a5a", "a5a")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 3 * 1e-5
    beta_sgd = 0.3
    lr_sag = 4 * 1e-5
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def a6a(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: a6a")

    dataset.load_data("DataSet/a6a", "a6a")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.4
    lr_sag = 1e-3
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def a7a(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: a7a")

    dataset.load_data("DataSet/a7a", "a7a")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.4
    lr_sag = 1e-3
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def a8a(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: a8a")

    dataset.load_data("DataSet/a8a", "a8a")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.4
    lr_sag = 1e-3
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def a9a(f, dataset, epochs):
    labels = ["LBFGS-B", "SGD", "SGD-M", "SAG", "SAGV2", "SAG-L", "SAGV2-L"]

    print("-------------------------------------------------------------------")
    print("DataSet: a9a")

    dataset.load_data("DataSet/a9a", "a9a")
    f.set_lamda(1 / dataset.data_train.shape[0])

    print("Samples: " + str(dataset.data_train.shape[0] + dataset.data_test.shape[0]) + "  features: " + str(
        dataset.data_train.shape[1]))
    dataset.print_balance()
    print("-------------------------------------------------------------------")

    lr_sgd = 1e-4
    beta_sgd = 0.4
    lr_sag = 1e-3
    lr_sagv2 = 1e-5

    test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2)


def test(f, dataset, epochs, labels, lr_sgd, beta_sgd, lr_sag, lr_sagv2):
    global list_w_LBFGSB
    list_w_LBFGSB = []

    x_time_all = []
    y_loss_all = []
    x_step_all = []

    w = np.ones(dataset.data_train.shape[1], dtype="float128")
    sol = sc.minimize(dataset.loss_function, w, method='L-BFGS-B',
                      jac=dataset.loss_gradient, callback=print_callback, options={'maxiter': epochs}).x
    print("Norm solution:", np.linalg.norm(sol))
    print("-------------------------------------------------------------------")
    list_w_LBFGSB = [w] + list_w_LBFGSB
    list_w_LBFGSB = fill_to_n_with_last_element(list_w_LBFGSB, epochs + 1)
    list_f_LBFGSB = []
    list_epoch = []
    for i in range(len(list_w_LBFGSB)):
        list_epoch.append(i)
        list_f_LBFGSB.append(dataset.loss_function(list_w_LBFGSB[i]))
    x_step_all.append(list_epoch)
    y_loss_all.append(list_f_LBFGSB)

    if "SGD" in labels:
        print("SGD Algorithms")
        _, x_step, y_loss, x_times, acc = exe.sgd(f, dataset, epochs, lr_sgd)
        plot(dataset.name + "/sgd_result.png", x_step, y_loss)
        plot(dataset.name + "/sgd_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SGD:", acc)

    if "SGD-M" in labels:
        print("SGD-M Algorithms")
        _, x_step, y_loss, x_times, acc = exe.sgd_momentum(f, dataset, epochs, lr_sgd, beta_sgd)
        plot(dataset.name + "/sgd_momentum_result.png", x_step, y_loss)
        plot(dataset.name + "/sgd_momentum_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SGD Momentum:", acc)

    if "SAG" in labels:
        print("SAG Algorithms")
        batch_size = int(dataset.data_train.shape[0] / 4)
        _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs, lr_sag)
        plot(dataset.name + "/sag_result.png", x_step, y_loss)
        plot(dataset.name + "/sag_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SAG:", acc)

    if "SAGV2" in labels:
        print("SAGV2 Algorithms")
        _, x_step, y_loss, x_times, acc = exe.sag_algorithm_v2(f, dataset, epochs, lr_sagv2)
        plot(dataset.name + "/sagV2_result.png", x_step, y_loss)
        plot(dataset.name + "/sagV2_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SAGV2:", acc)

    if "SAG-L" in labels:
        print("SAG-L Algorithms")
        _, x_step, y_loss, x_times, acc = exe.sag_algorithm(f, dataset, epochs)
        plot(dataset.name + "/sag_L_result.png", x_step, y_loss)
        plot(dataset.name + "/sag_L_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SAG-LS:", acc)

    if "SAGV2-L" in labels:
        print("SAGV2-L Algorithms")
        _, x_step, y_loss, x_times, acc = exe.sag_algorithm_v2(f, dataset, epochs)
        plot(dataset.name + "/sagV2-L_result.png", x_step, y_loss)
        plot(dataset.name + "/sagV2-L_result_time.png", x_times, y_loss)
        x_time_all.append(x_times)
        y_loss_all.append(y_loss)
        x_step_all.append(x_step)
        print("Accuracy SAGV2-LS:", acc)

    plot_full(dataset.name + "/full_step_result_last_run.png", labels, x_step_all, y_loss_all, "Epochs")
    plot_full(dataset.name + "/full_time_result_last_run.png", labels, x_time_all, y_loss_all, "Time")


if __name__ == '__main__':
    threshold = 0.5
    split = 0.8
    epochs = 50
    np.random.seed(17)

    f = Function(threshold)
    dataset = DataSet(split)
    exe = Solver()

    # skin_nonskin(f, dataset, epochs)  # OK DA VEDERE IPER-LENTO
    # german_numer_scale(f, dataset, epochs)  # OK 100
    # phishing(f, dataset, epochs)  # OK 300
    # ijcnn1(f, dataset, epochs)  # ok
    # cod_rna(f, dataset, epochs)  # OK 500

    # australian(f, dataset, epochs)  # OK 100 seed(17)
    # a5a(f, dataset, epochs)  # OK 500
    # a6a(f, dataset, epochs)  # DA VEDERE IPER-LENTO
    # a7a(f, dataset, epochs) # DA VEDERE IPER-LENTO
    # a8a(f, dataset, epochs) # DA VEDERE IPER-LENTO
    # a9a(f, dataset, epochs) # DA VEDERE IPER-LENTO
