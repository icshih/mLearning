import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_target_function():
    x = ss.uniform.rvs(-1, scale=2, size=2)
    y = ss.uniform.rvs(-1, scale=2, size=2)
    a = np.array([[x[0], 1], [x[1], 1]])
    b = np.array([y[0], y[1]])
    param = np.linalg.inv(a).dot(b)
    return param


def get_data(target_function, num_data):
    b = np.empty([num_data, 3])
    for i in range(num_data):
        x = ss.uniform.rvs(-1, scale=2, size=1)
        y = ss.uniform.rvs(-1, scale=2, size=1)
        ty = target_function[0]*x + target_function[1]
        if (y - ty) >= 0:
            b[i] = [x, y, 1]
        else:
            b[i] = [x, y, -1]
    temp = pd.DataFrame(b)
    temp.columns = ['x', 'y', 'b']
    return temp


def linear_regression(data_set):
    data_set['z'] = 1.0
    x = np.asarray(data_set[['z', 'x', 'y']])
    x_trans = x.T
    b_ = np.asarray(data_set[['b']])
    inv = np.linalg.inv(x_trans.dot(x))
    x_pseudo_inv = inv.dot(x_trans)
    w = x_pseudo_inv.dot(b_)
    return w


def get_hypothesis_line(x, weight):
    return -1.0*(weight[1]/weight[2])*x - 1.0*(weight[0]/weight[2])


def find_misclassified(data_set, weight):
    y_ = -1.0*(weight[1]/weight[2])*data_set['x'] - 1.0*(weight[0]/weight[2])
    return dataset[(data_set['y'] - y_)*data_set['b'] < 0]


if __name__ == '__main__':
    """This is an implementation of the perceptron learning algorithm based on the homework of Learning From Data."""

    NUM_DATA = 100
    NUM_FRESH_DATA = 1000

    line = np.arange(-1.0, 1.1, 0.1)

    mis_in = np.empty(1000)
    mis_out = np.empty(1000)
    for i in range(1000):
        target = get_target_function()
        dataset = get_data(target, NUM_DATA)
        data_fresh = get_data(target, NUM_FRESH_DATA)
        weight = linear_regression(dataset)
        mis_clas_in = find_misclassified(dataset, weight)
        mis_clas_out = find_misclassified(data_fresh, weight)
        mis_in[i] = len(mis_clas_in)
        mis_out[i] = len(mis_clas_out)

    print(mis_in.mean())
    print(mis_out.mean())

    # positive = dataset[dataset['b'] > 0.0]
    # negative = dataset[dataset['b'] < 0.0]
    #
    # f = target[0]*line + target[1]
    # g = get_hypothesis_line(line, weight)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlim(-1.0, 1.0)
    # ax.set_ylim(-1.0, 1.0)
    # ax.plot(positive['x'], positive['y'], 'bo')
    # ax.plot(negative['x'], negative['y'], 'bx')
    # ax.plot(line, f, 'b-')
    # ax.plot(line, g, 'y-')
    # plt.show()