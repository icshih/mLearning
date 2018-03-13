import multiprocessing
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_target_function():
    x = ss.uniform.rvs(-1, scale=2, size=2)
    y = ss.uniform.rvs(-1, scale=2, size=2)
    A = np.array([[x[0], 1], [x[1], 1]])
    b = np.array([y[0], y[1]])
    param = np.linalg.inv(A).dot(b)
    return param


def get_hypothesis_line(x, w0, w1, w2):
    return -1.0*(w1/w2)*x - 1.0*(w0/w2)


def get_data(target):
    x = ss.uniform.rvs(-1, scale=2, size=1)
    y = ss.uniform.rvs(-1, scale=2, size=1)
    ty = target[0]*x + target[1]
    if (y - ty) >= 0:
        b = 1
    else:
        b = -1
    return x, y, b


def get_dataset(target, num_data):
    data = np.zeros([num_data, 3])
    for i in range(num_data):
        x, y, b = get_data(target)
        data[i] = [x, y, b]
    dataset = pd.DataFrame(data)
    dataset.columns = ['x', 'y', 'b']
    return dataset


def pla(dataset, w0, w1, w2):
    x = list()
    y = list()
    b = list()
    for i in range(len(dataset)):
        temp = dataset.iloc[i]
        weight = np.array([[w0, w1, w2]])
        x_ = np.array([[1, temp['x'], temp['y']]])
        h = np.sum(weight*x_)
        # not match
        if (h*temp['b'] <= 0):
            x.append(temp['x'])
            y.append(temp['y'])
            b.append(temp['b'])
    out = {'x': x, 'y': y, 'b': b}
    return pd.DataFrame(out)


def update_weight(misclas, w0, w1, w2):
    sample = misclas.sample(n=1)
    w0 = w0 + 1.0*sample['b']
    w1 = w1 + sample['x']*sample['b']
    w2 = w2 + sample['y']*sample['b']
    return w0.values[0], w1.values[0], w2.values[0]


def converge(data_set, num_iter):
    W0 = list()
    W1 = list()
    W2 = list()
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    for i in range(num_iter):
        W0.append(w0)
        W1.append(w1)
        W2.append(w2)
        output = pla(data_set, w0, w1, w2)
        if len(output) == 0:
            print('After {0} iterations, all points are classified'.format(i+1))
            break
        else:
            w0, w1, w2 = update_weight(output, w0, w1, w2)
    if (i+1) == num_iter:
        print('After {0} iterations, no solution is found'.format(num_iter))
    out = {'w0': W0, 'w1': W1, 'w2': W2}
    return pd.DataFrame(out)


if __name__ == '__main__':
    """This is an implementation of the perceptron learning algorithm based on the homework of Learning From Data."""

    NUM_DATA = 100
    NUM_ITER = 1000
    NUM_RUN = 10

    line = np.arange(-1.0, 1.1, 0.1)

    # with multiprocessing.Pool(1) as pool:
    #     output = pool.map(worker, [x for x in range(num_run)])
    #
    # result = pd.Series(output)
    # print(result[result != 1000].mean())

    target = get_target_function()
    dataset = get_dataset(target, NUM_DATA)
    weight = converge(dataset, NUM_ITER)

    positive = dataset[dataset['b'] > 0.0]
    negative = dataset[dataset['b'] < 0.0]

    f = target[0]*line + target[1]
    g_w = weight.tail(1)
    g = get_hypothesis_line(line, g_w['w0'].values[0], g_w['w1'].values[0], g_w['w2'].values[0])
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.plot(positive['x'], positive['y'], 'bo')
    ax.plot(negative['x'], negative['y'], 'bx')
    ax.plot(line, f, 'b-')
    ax.plot(line, g, 'y-')
    plt.show()