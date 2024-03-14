import numpy as np


class Function(object):
    def __init__(self, threshold):
        self.lamda = 0
        self.threshold = threshold

    def set_lamda(self, lamda):
        self.lamda = lamda

    """
    This method implements the function loss of the logist regression for a single f_i.
    """

    def loss_function_f(self, x, y, w):
        return np.log(1 + (np.exp(-y * np.dot(w, x))))

    """
    This method implements the gradient of the logistic regression for a single f_i.
    """

    def loss_gradient_f(self, x, y, w):
        return (- y * x) / (1 + np.exp(y * np.dot(x, w)))

    """
    This method implements the function loss of the logist regression with a strongly-convex regularizer, 
    the squared l2 -norm, and return the value of the loss in a specific point W.
    """

    def loss_function(self, x, y, w):
        reg_term = self.lamda / 2 * np.linalg.norm(w) ** 2
        loss_term = np.mean(np.logaddexp(0, -y * np.dot(x, w)))
        return reg_term + loss_term

    """
    This method implements the gradient of the function loss with numpy function.
    """
    def loss_gradient_2(self, x, y, w):
        r = np.multiply(-y, self.sigmoid(np.multiply(-y, np.dot(x, w))))
        return np.matmul(x.T, r) + self.lamda * w

    """
    This method implements the gradient of the function loss with a for.
    """
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

    """
    This method returns the accuracy of the logistic regression with a specific W 
    (obtained minimizing the loss function, and passed by argument w) in a set of data passed by arguments (x_t, y_t)
    """

    def testing(self, x_t, y_t, w):
        count = 0
        for i in range(x_t.shape[0]):
            if y_t[i] == self.predict(w, x_t[i]):
                count = count + 1
        return (count / x_t.shape[0]) * 100
