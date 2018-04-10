import scipy.stats as ss
import numpy as np
import pandas as pd


def get_target_function():
    x1 = ss.uniform.rvs(-1, scale=2, size=1000)
    x2 = ss.uniform.rvs(-1, scale=2, size=1000)
    b = x1*x1 + x2*x2 - 0.6
    data = pd.DataFrame(np.asarray([x1, x2, b]).T)
    data.columns = ['x', 'y', 'b']
    data.b[data.b < 0] = -1
    data.b[data.b >= 0] = 1
    return data


def set_noise(dataset, size_percent):
    """set a percentage of data with noise, the column b is modified."""
    samp = dataset.sample(n=int(len(dataset)*size_percent))
    dataset.b[samp.index]*-1.0
    return dataset


def linear_regression(data_set):
    """In the form of (1, x, y)"""
    data_set['z'] = 1.0
    x = np.asarray(data_set[['z', 'x', 'y']])
    x_trans = x.T
    inv = np.linalg.inv(x_trans.dot(x))
    x_pseudo_inv = inv.dot(x_trans)
    b_ = np.asarray(data_set[['b']])
    w = x_pseudo_inv.dot(b_)
    return w


def linear_regression2(data_set):
    """In the form of (1, x, y, xy, xx, yy)"""
    data_set['z'] = 1.0
    data_set['xy'] = data_set['x']*data_set['y']
    data_set['xx'] = data_set['x']*data_set['x']
    data_set['yy'] = data_set['y']*data_set['y']
    x = np.asarray(data_set[['z', 'x', 'y', 'xy', 'xx', 'yy']])
    x_trans = x.T
    inv = np.linalg.inv(x_trans.dot(x))
    x_pseudo_inv = inv.dot(x_trans)
    b_ = np.asarray(data_set[['b']])
    w = x_pseudo_inv.dot(b_)
    return w


def find_misclassified(data_set, weight):
    y_ = 1.0*weight[0] + data_set['x']*weight[1] + data_set['y']*weight[2]
    return data_set[y_*data_set['b'] < 0]


def find_misclassified2(data_set, weight):
    y_ = 1.0*weight[0] + data_set['x']*weight[1] + data_set['y']*weight[2] + data_set['xy']*weight[3] + data_set['xx']*weight[4] + data_set['yy']*weight[5]
    return data_set[y_*data_set['b'] < 0]


if __name__ == '__main__':
    """This is an implementation of the nonlinear transfomation based on the homework of Learning From Data."""

    weights = np.zeros((1000, 6))
    mis_clasi = np.zeros(1000)
    mis_clasi2 = np.zeros(1000)
    for i in range(1000):
        data = get_target_function()
        set_noise(data, 0.1)
        weight = linear_regression(data)
        weight2 = linear_regression2(data)
        weights[i] = [weight2[0], weight2[1], weight2[2], weight2[3], weight2[4], weight2[5]]
        mis_data = find_misclassified(data, weight)
        mis_data2 = find_misclassified2(data, weight2)
        mis_clasi[i] = len(mis_data)
        mis_clasi2[i] = len(mis_data2)
    print('Average misclassified (non-transformed): {}'.format(mis_clasi.mean()))
    print('Average misclassified (transformed): {}'.format(mis_clasi2.mean()))

    W = pd.DataFrame(weights)
    W.columns = ['a', 'b', 'c', 'd', 'e', 'f']
    print(W['a'].mean())
    print(W['b'].mean())
    print(W['c'].mean())
    print(W['d'].mean())
    print(W['e'].mean())
    print(W['f'].mean())
