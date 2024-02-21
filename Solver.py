import numpy as np
from tqdm import tqdm
import time


class Solver:
    def __init__(self, precision):
        self.precision = precision

    def sgd(self, f, dataset, epochs, learn_rate, batch_size):
        rng = np.random.default_rng()
        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        w = np.zeros(dataset.data_train.shape[1], dtype="float128")
        n_obs = x.shape[0]
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
        learn_rate = np.array(learn_rate)
        x_plt, y_plt, x_times = [], [], []

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            rng.shuffle(xy)
            for start in range(0, n_obs, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
                y_batch = np.squeeze(y_batch, axis=1)
                direction = f.loss_gradient(x_batch, y_batch, w)
                w = w - learn_rate * direction

            print("Train loss: " + str(f.loss_function(x, y, w)))
            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

            if (epoch + 1) % 10 == 0:
                acc = f.testing(dataset.data_test, dataset.labels_test, w)
                print("Accuracy epoch " + str(epoch + 1) + ": " + str(acc) + "%")

            if x_times:
                last_element = x_times[-1]
            else:
                last_element = 0
            x_times.append((time.time() - start_time) + last_element)

        return w, x_plt, y_plt, x_times

    def sgd_momentum(self, f, dataset, epochs, learn_rate, batch_size, beta):

        rng = np.random.default_rng()
        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        w = np.zeros(dataset.data_train.shape[1], dtype="float128")
        n_obs = x.shape[0]
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
        beta = np.array(beta, dtype="float128")
        learn_rate = np.array(learn_rate, dtype="float128")
        x_plt, y_plt, x_times = [], [], []

        diff = 0
        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            rng.shuffle(xy)
            for start in range(0, n_obs, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
                y_batch = np.squeeze(y_batch, axis=1)

                direction = f.loss_gradient(x_batch, y_batch, w)

                diff = beta * diff - learn_rate * direction
                w += diff

            print("Train loss: " + str(f.loss_function(x, y, w)))
            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

            if (epoch + 1) % 10 == 0:
                acc = f.testing(dataset.data_test, dataset.labels_test, w)
                print("Accuracy epoch " + str(epoch + 1) + ": " + str(acc) + "%")

            if x_times:
                last_element = x_times[-1]
            else:
                last_element = 0
            x_times.append((time.time() - start_time) + last_element)

        return w, x_plt, y_plt, x_times
