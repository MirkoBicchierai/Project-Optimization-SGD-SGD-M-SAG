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
        w_memory_momentum = np.zeros(dataset.data_train.shape[1], dtype="float128")
        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            rng.shuffle(xy)
            for start in range(0, n_obs, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
                y_batch = np.squeeze(y_batch, axis=1)

                direction = f.loss_gradient(x_batch, y_batch, w)
                diff = beta * (w - w_memory_momentum) - learn_rate * direction
                w_memory_momentum = w
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

        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            idx = np.random.randint(0, n_samples)
            g = f.loss_gradient(X[idx:idx + 1], y[idx:idx + 1], w)
            d = d - memory[idx] + g
            memory[idx] = d
            w -= (learn_rate / n_samples) * d

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

        gradient_memory = np.zeros((n_samples, n_features))
        average_gradient = np.zeros(n_features)

        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            indices = np.random.choice(n_samples, batch_size, replace=False)

            X_batch, y_batch = X[indices], y[indices]
            batch_gradient = f.loss_gradient(X_batch, y_batch, w)  # Calcolo il gradiente per il mini batch

            # Aggiorno la media dei gradienti per ogni indice generato

            for i in indices:
                old_gradient = gradient_memory[i]
                average_gradient += batch_gradient - old_gradient
                gradient_memory[i] = batch_gradient

            w -= (learn_rate / n_samples) * average_gradient  # Update di w_k+1

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

        list_hit = np.zeros(n_samples)

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(X, y, w))

        for epoch in tqdm(range(epochs)):
            start_time = time.time()

            idx = np.random.randint(0, n_samples)

            list_hit[idx] += 1
            eq = np.sum(list_hit)

            g = f.loss_gradient(X[idx:idx + 1], y[idx:idx + 1], w)
            d = d + - memory[idx] + g  # eq
            memory[idx] = d

            L = self.lipschitzEstimate(f, X[idx:idx + 1], y[idx:idx + 1], w)
            w -= ((1 / L) / n_samples) * d

            x_plt.append(epoch)
            y_plt.append(f.loss_function(X, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    def lipschitzEstimate(self, f, X, y, w):
        l_lip = 1
        max_iter = 1000
        c = 0
        old_loss = f.loss_function(X, y, w)
        grad = f.loss_gradient(X, y, w)
        norm = pow(np.linalg.norm(grad), 2)
        # if norm > pow(10, -8):
        new_w = w - (1 / l_lip) * grad
        new_loss = f.loss_function(X, y, new_w)
        while new_loss > old_loss - 1 / (2 * l_lip) * norm and c < max_iter:
            l_lip = l_lip * (1 / pow(2, -(1 / X.shape[0])))
            c = c + 1

        return l_lip

    def sag_algorithm_LS_2(self, f, dataset, epochs, learning_rate="L-LS"):

        X, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype="float128")

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(X, y, w))

        memory = np.zeros((n_samples, n_features))
        d = np.mean(memory, axis=0)
        t = np.zeros(n_features)
        agg = np.zeros(n_features)
        c = np.zeros(n_samples)
        m = 0
        w_memory = np.zeros((n_features, epochs))

        for epoch in tqdm(range(epochs)):
            start_time = time.time()
            idx = np.random.randint(0, n_samples)

            if c[idx] == 0:
                m = m + 1
                c[idx] = 1

            if learning_rate == "L-LS":
                learning_rate = 1 / (16*n_samples*self.lipschitzEstimate(f, X[idx:idx + 1], y[idx:idx + 1], w))

            for j in range(len(X[idx])):
                if X[idx][j] == 0:
                    t[j] = t[j] + 1
                    agg[j] = -1
                else:
                    if t[j] == 0:
                        agg[j] = 0  # Aggiorno normale
                    else:
                        agg[j] = t[j]  # Aggiorno All'iterzaione t[j]
                        t[j] = 0

            g = f.loss_gradient(X[idx:idx + 1], y[idx:idx + 1], w)
            d = d + - memory[idx] + g
            memory[idx] = d

            for var in range(len(agg)):
                if agg[var] == 0:
                    w[var] = w[var] - (learning_rate/ m) * d[var]
                if agg[var] > 0:
                    old_iter = epoch - agg[var]
                    w[var] = w_memory[var, int(old_iter)] - ((learning_rate * agg[var])/ m) * d[var]
                    agg[var] = 0  # reset

            w_memory[:, epoch] = w  # traccio w

            x_plt.append(epoch)
            y_plt.append(f.loss_function(X, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)
