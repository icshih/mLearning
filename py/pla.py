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


def get_hypothesis_line():
    np.arrange()


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
    num_data = 10
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


def converge(data_set):
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    for i in range(1000):
        output = pla(data_set, w0, w1, w2)
        print(w0, w1, w2, len(output))
        if (len(output) == 0):
            print('After {} iterations, all points are classfied'.format(i+1))
            break
        else:
            w0, w1, w2 = update_weight(output, w0, w1, w2)
    if (i+1) == 1000:
        print('After 1000 iterations, no solution is found')
    return i+1


def worker(iter):
    t = get_target_function()
    d = get_dataset(t, num_data)
    num_iter = converge(d)
    return num_iter


def test_worker(iter):
    return iter


if __name__ == '__main__':

    num_data = 10
    num_run = 10

    line = np.arange(-1.0, 1.1, 0.1)

    # with multiprocessing.Pool(1) as pool:
    #     output = pool.map(worker, [x for x in range(num_run)])
    #
    # result = pd.Series(output)
    # print(result[result != 1000].mean())

    target = get_target_function()
    dataset = get_dataset(target, num_data)
    postive = dataset[dataset['b'] > 0.0]
    negative = dataset[dataset['b'] < 0.0]

    converge(dataset)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.plot(postive['x'], postive['y'], 'ro')
    ax.plot(negative['x'], negative['y'], 'go')
    ax.plot(line, target[0]*line + target[1], 'b-')
    ax.plot(line, 0.0*line - 0.0, 'y-')
    plt.show()