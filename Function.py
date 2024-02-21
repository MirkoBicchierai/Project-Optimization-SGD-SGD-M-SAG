import numpy as np


class Function(object):
    def __init__(self):
        self.lamda = 0.8
        self.threshold = 0.5

    def loss_function(self, x, y, w):
        reg_term = self.lamda / 2 * np.linalg.norm(w) ** 2
        loss_term = np.mean(np.logaddexp(0, -y * np.dot(x, w)))
        total_value = reg_term + loss_term
        return total_value

    def loss_gradient(self, x, y, w):
        n = len(x)
        reg_gradient = self.lamda * w
        logistic_gradient = np.zeros_like(w)
        for i in range(n):
            exponent = y[i] * np.dot(x[i], w)
            logistic_gradient += - y[i] * x[i] / (1 + np.exp(exponent))

        gradient = reg_gradient + logistic_gradient / n
        return gradient

    def sigmoid(self, x):
        x = np.array(x, np.float128)
        return 1 / (1 + np.exp(-x))

    def predict(self, w, x):
        if self.sigmoid(np.dot(w, x)) >= self.threshold:
            return 1
        else:
            return -1

    def testing(self, x_t, y_t, w):
        count = 0
        for i in range(x_t.shape[0]):
            if y_t == self.predict(w, x_t[i]):
                count = count + 1
        return (count / x_t.shape[0]) * 100
