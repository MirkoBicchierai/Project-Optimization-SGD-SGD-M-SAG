import numpy as np
from tqdm import tqdm
from DataSet import DataSet
from numpy import linalg as la

threshold = 0.5
lambda_reg = 1


def sigmoid(x):
    x = np.array(x, np.float128)
    return 1 / (1 + np.exp(-x))


def testing(dataset, weights):
    count = 0
    for i in range(dataset.data_test.shape[0]):
        prediction = predict(weights, dataset.data_test[i])
        if dataset.labels_test[i] == prediction:
            count = count + 1
    return (count / dataset.data_test.shape[0]) * 100


def predict(weights, data):
    if sigmoid(np.dot(weights, data)) >= threshold:
        return 1
    else:
        return -1


def gradient(X, y, weights):
    r = np.multiply(-y, sigmoid(np.multiply(-y, np.dot(X, weights))))
    return (np.matmul(X.T, r) * (1 / X.shape[0])) + lambda_reg * weights


def compute_log_loss(X, y, w):
    return ((1 / X.shape[0]) * np.sum(
        np.log(1 + np.exp(-y * np.dot(X, w)))) + lambda_reg / 2 * la.norm(
        w) ** 2)

def loss_gradient(X, y, w):
    n = len(X)
    reg_gradient = lambda_reg * w
    logistic_gradient = np.zeros_like(w)
    for i in range(n):
        exponent = y[i] * np.dot(X[i], w)
        logistic_gradient += - y[i] * X[i] / (1 + np.exp(exponent))

    gradient = reg_gradient + logistic_gradient/n
    return gradient


def calculate_function(X, y, w):
    reg_term = lambda_reg / 2 * np.linalg.norm(w) ** 2
    loss_term = np.mean(np.logaddexp(0, -y * np.dot(X, w)))
    total_value = reg_term + loss_term
    return total_value

dataset = DataSet("DataSet/new_australian.csv", ",", 0)

x = dataset.data_train
y = dataset.labels_train
x, y = np.array(x, dtype="float128"), np.array(y, dtype="float128")
w = np.zeros(dataset.data_train.shape[1], dtype="float128")

learn_rate = 1e-4
batch_size = 1 #x.shape[0]
n_iter = 50000
n_obs = x.shape[0]
xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

# Initializing the random number generator
rng = np.random.default_rng()
learn_rate = np.array(learn_rate)
loss=0
for ss in tqdm(range(n_iter)):

    rng.shuffle(xy)
    # Performing minibatch moves
    for start in range(0, n_obs, batch_size):
        stop = start + batch_size
        x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
        y_batch = np.squeeze(y_batch, axis=1)

        grad = loss_gradient(x_batch, y_batch, w)
        diff = -learn_rate * grad
        w += diff

    loss += calculate_function(x, y, w)
    if (ss + 1) % 100 == 0:
        print("Log loss: " + str(loss/100))
        loss = 0

    if (ss + 1) % 200 == 0:
        acc = testing(dataset, w)
        print("Accuracy: " + str(acc) + "%")
