import numpy as np
from tqdm import tqdm
import time


class Solver:
    """
        def sgd(self, f, dataset, epochs, learn_rate):

        Method to run SGD Algorithm to found the optimal value of W.
        This method implements the standard SGD Algorithm with constant step size, passed as parameters of the function.

    """

    def sgd(self, f, dataset, epochs, learn_rate):
        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        # w = np.zeros(dataset.data_train.shape[1], dtype="float128")
        w = np.ones(dataset.data_train.shape[1], dtype="float128")

        n_obs = x.shape[0]
        learn_rate = np.array(learn_rate)
        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

        v_index = list(range(n_obs))

        for epoch in tqdm(range(epochs)):
            np.random.shuffle(v_index)
            start_time = time.time()
            for i in v_index:
                direction = f.loss_gradient(x[i:i + 1], y[i:i + 1], w)
                w = w - learn_rate * direction

            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    """
        sgd_momentum(self, f, dataset, epochs, learn_rate, beta)::

        Method to run SGD Algorithm with momentum to found the optimal value of W. 
        This method implements the SGD Algorithm with constant step size and constant momentum term,
        both passed as parameters of the function (learn_rate, beta).
        
    """

    def sgd_momentum(self, f, dataset, epochs, learn_rate, beta):

        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        w = np.ones(dataset.data_train.shape[1], dtype="float128")
        n_obs = x.shape[0]
        beta = np.array(beta, dtype="float128")
        learn_rate = np.array(learn_rate, dtype="float128")
        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

        v_index = list(range(n_obs))
        w_memory_momentum = w
        for epoch in tqdm(range(epochs)):
            np.random.shuffle(v_index)
            start_time = time.time()
            for i in v_index:
                direction = f.loss_gradient(x[i:i + 1], y[i:i + 1], w)
                diff = beta * (w - w_memory_momentum) - learn_rate * direction
                w_memory_momentum = w
                w = w + diff

            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    """
        sag_algorithm_v2(self, f, dataset, epochs, lr="L-LS"):

        Method to run SAG Algorithm to found the optimal value of W. 
        This method implements the standard SAG Algorithm.
        
        If nothing is passed as the "lr" (learning rate) parameter, a learning rate of 1/16L will be used, 
        with L being the Lipschitz constant, found using a line search.
    """

    def sag_algorithm(self, f, dataset, epochs, learn_rate="L-LS"):

        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = x.shape

        w = np.ones(n_features, dtype="float128")
        memory = np.zeros((n_samples, n_features))
        d = np.mean(memory, axis=0)

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

        v_index = list(range(n_samples))

        for epoch in tqdm(range(epochs)):
            np.random.shuffle(v_index)
            start_time = time.time()
            for i in v_index:
                g = f.loss_gradient(x[i:i + 1], y[i:i + 1], w)
                d = d - memory[i] + g
                memory[i] = g
                if learn_rate == "L-LS":
                    l = self.lipschitz_estimate(f, x[i:i + 1], y[i:i + 1], w)
                    lr = (1 / (16 * l))
                else:
                    lr = learn_rate
                w -= (lr / n_samples) * d

            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))
            x_times.append((time.time() - start_time) + x_times[-1])

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    """
        sag_algorithm_v2(self, f, dataset, epochs, lr="L-LS"):
        
        Method to run SAG Algorithm to found the optimal value of W. 
        This method implements SAG Algorithm with all the "trick" described in the paper at section 4.
        If nothing is passed as the "lr" (learning rate) parameter, a learning rate of 1/16L will be used, 
        with L being the Lipschitz constant, found using a line search.
    """

    def sag_algorithm_v2(self, f, dataset, epochs, lr="L-LS"):

        x, y = np.array(dataset.data_train, dtype="float128"), np.array(dataset.labels_train, dtype="float128")
        n_samples, n_features = x.shape
        w = np.ones(n_features, dtype="float128")

        x_plt, y_plt, x_times = [], [], []
        x_plt.append(0)
        x_times.append(0)
        y_plt.append(f.loss_function(x, y, w))

        memory = np.zeros((n_samples, n_features))
        d = np.mean(memory, axis=0)

        v_index = list(range(n_samples))

        for epoch in tqdm(range(epochs)):

            w_memory = np.zeros((n_features, n_samples))
            t = np.zeros(n_features)
            agg = np.zeros(n_features)
            c = np.zeros(n_samples)
            m = 0
            np.random.shuffle(v_index)
            counter = 0

            start_time = time.time()
            for idx in v_index:
                if c[idx] == 0:
                    m = m + 1
                    c[idx] = 1

                if lr == "L-LS":
                    l = self.lipschitz_estimate(f, x[idx:idx + 1], y[idx:idx + 1], w)
                    learning_rate = (1 / (16 * l))
                else:
                    learning_rate = lr

                for j in range(len(x[idx])):
                    if x[idx][j] == 0:
                        t[j] = t[j] + 1
                        agg[j] = -1
                    else:
                        if t[j] == 0:
                            agg[j] = 0
                        else:
                            agg[j] = t[j]
                            t[j] = 0

                g = f.loss_gradient(x[idx:idx + 1], y[idx:idx + 1], w)
                d = d - memory[idx] + g
                memory[idx] = g

                for var in range(len(agg)):
                    if agg[var] == 0:
                        w[var] = w[var] - (learning_rate / m) * d[var]
                    if agg[var] > 0:
                        old_iter = counter - agg[var]
                        w[var] = w_memory[var, int(old_iter)] - ((learning_rate * agg[var]) / m) * d[var]
                        agg[var] = 0

                w_memory[:, counter] = w
                counter = counter + 1

            x_times.append((time.time() - start_time) + x_times[-1])
            x_plt.append(epoch)
            y_plt.append(f.loss_function(x, y, w))

        return w, x_plt, y_plt, x_times, f.testing(dataset.data_test, dataset.labels_test, w)

    """
        lipschitz_estimate(self, f, x, y, w):
        
        Method to do the line search to found the lipschitz constant L, starting by initial value of L = 1
        This line search is described in the paper at section 4.6.
    """

    def lipschitz_estimate(self, f, x, y, w):
        l_lip = 1
        max_iter = 100
        old_loss = f.loss_function(x, y, w)
        norm = pow(np.linalg.norm(old_loss), 2)
        for sus in range(max_iter):
            new_w = w - (1 / l_lip) * old_loss
            new_loss = f.loss_function(x, y, new_w)
            if new_loss <= old_loss - (1 / (2 * l_lip)) * norm:
                break
            l_lip = l_lip * 2
        return l_lip
