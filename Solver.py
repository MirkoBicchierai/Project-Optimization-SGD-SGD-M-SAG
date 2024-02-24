import numpy as np
from tqdm import tqdm
import time


class Solver:

    def sgd(self, f, dataset, epochs, learn_rate, batch_size):
        rng = np.random.default_rng()
        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        w = np.zeros(dataset.data_train.shape[1], dtype="float128")
        n_obs = x.shape[0]
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
        learn_rate = np.array(learn_rate)
        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            rng.shuffle(xy)
            for start in range(0, n_obs, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
                y_batch = np.squeeze(y_batch, axis=1)
                direction = f.loss_gradient(x_batch, y_batch, w)
                w = w - learn_rate * direction

            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    def sgd_momentum(self, f, dataset, epochs, learn_rate, batch_size, beta):

        rng = np.random.default_rng()
        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        w = np.zeros(dataset.data_train.shape[1], dtype="float128")
        n_obs = x.shape[0]
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
        beta = np.array(beta, dtype="float128")
        learn_rate = np.array(learn_rate, dtype="float128")
        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

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

            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    def sag_algorithm(self, f, dataset, epochs, learn_rate):

        X, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype="float128")
        memory = np.zeros((n_samples, n_features))
        d = np.mean(memory, axis=0)

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(X, y, w))

        list_gen = []
        # L = 100
        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            idx = np.random.randint(0, n_samples)
            g = f.loss_gradient(X[idx:idx + 1], y[idx:idx + 1], w)
            d = d - memory[idx] + g
            memory[idx] = d
            w -= (learn_rate / n_samples) * d  # len(list_gen)

            x_plt.append(epoch)
            y_plt.append(f.loss_function(X, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    def sag_algorithm_B(self, f, dataset, epochs, learn_rate, batch_size):

        X, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype="float128")

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(X, y, w))

        gradient_memory = np.zeros((n_samples, n_features))  # Store gradients for each sample
        average_gradient = np.zeros(n_features)  # Initialize the average gradient

        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            indices = np.random.choice(n_samples, batch_size, replace=False)  # Randomly pick a mini-batch
            X_batch, y_batch = X[indices], y[indices]
            batch_gradient = f.loss_gradient(X_batch, y_batch, w)  # Compute gradient for the mini-batch

            # Update the running average gradient

            for i in indices:
                old_gradient = gradient_memory[i]
                average_gradient += (batch_gradient - old_gradient) / n_samples
                gradient_memory[i] = batch_gradient

            w -= learn_rate * average_gradient  # Update the parameters

            x_plt.append(epoch)
            y_plt.append(f.loss_function(X, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    def sag_algorithm_LS(self, f, dataset, epochs):

        X, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype="float128")
        memory = np.zeros((n_samples, n_features))
        d = np.mean(memory, axis=0)

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(X, y, w))

        list_gen = []
        L = 1
        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            idx = np.random.randint(0, n_samples)
            # if idx not in list_gen:
            #     list_gen.append(idx)
            g = f.loss_gradient(X[idx:idx + 1], y[idx:idx + 1], w)
            d = d - memory[idx] + g
            memory[idx] = d
            #w -= (learn_rate / n_samples) * d  # len(list_gen)

            if lipschitzEstimate(f, L, epoch, X[idx:idx + 1], y[idx:idx + 1], w):
                L = L * 2
            w -= (1 / (pow(L, epoch))) * d  # len(list_gen)

            x_plt.append(epoch)
            y_plt.append(f.loss_function(X, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)


def lipschitzEstimate(f, L, k, X, y, w):
    old_loss = f.loss_function(X, y, w)
    grad = f.loss_gradient(X, y, w)
    norm = pow(np.linalg.norm(grad), 2)
    if norm > pow(10, -8):
        new_w = w - (1 / pow(L, k)) * grad
        new_loss = f.loss_function(X, y, new_w)
        if new_loss >= old_loss - 1 / (2 * pow(L, k)) * norm:
            return False
        else:
            return True
    else:
        return False
