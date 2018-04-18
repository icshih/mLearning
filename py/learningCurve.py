import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_target_function(num_of_data):
    x = ss.uniform.rvs(-1, scale=2, size=num_of_data)
    y = np.sin(np.pi*x) + ss.norm.rvs(loc=0, scale=0.01, size=num_of_data)
    temp = pd.DataFrame(np.asarray([x, y]).T)
    temp.columns = ['x', 'y']
    return temp


def hypothesis0(data_set):
    m = np.asarray(data_set[['x']])
    m_trans = m.T
    m_pesudo = np.linalg.inv(m_trans.dot(m)).dot(m_trans)
    w = m_pesudo.dot(np.asarray(data_set[['y']]))
    return w


def hypothesis0_(data_set):
    y = data_set[['y']]
    return y.mean()


def hypothesis1(data_set):
    data_set['z'] = 1
    m = np.asarray(data_set[['z', 'x']])
    m_trans = m.T
    m_pesudo = np.linalg.inv(m_trans.dot(m)).dot(m_trans)
    w = m_pesudo.dot(np.asarray(data_set[['y']]))
    weight = pd.DataFrame(w.T)
    weight.columns = ['w0', 'w1']
    return weight


def get_hypothesis0(target_function_data, hypothesis_weight):
    x = target_function_data['x']
    y = np.zeros(len(target_function_data))
    y[:] = hypothesis_weight
    temp = pd.DataFrame(np.asarray([x, y]).T)
    temp.columns = ['x', 'y']
    return temp


def get_hypothesis1(target_function_data, hypothesis_weight):
    w0 = hypothesis_weight['w0'].get_values()
    w1 = hypothesis_weight['w1'].get_values()
    x = target_function_data['x'].get_values()
    y = 1.0*w0 + w1 * x
    temp = pd.DataFrame(np.asarray([x, y]).T)
    temp.columns = ['x', 'y']
    return temp


def sample_error_vector(sample_data_set, weight):
    vector = list()
    w1 = weight['w1']
    for k, v in enumerate(sample_data_set.get_values()):
        vector.append((w1 * v[0]) - v[1])
    return vector


def square_mean_error(error_vector):
    return np.sum(np.power(error_vector, 2))/len(error_vector)


if __name__ == '__main__':
    """This is an exercise of bias and variance based on the homework of Learning From Data."""

    tf_data = get_target_function(200)
    in_sample = get_target_function(2)
    print(in_sample)
    w_0_ = hypothesis0(in_sample)
    w_0 = hypothesis0_(in_sample)
    w_1 = hypothesis1(in_sample)
    print(w_0_)
    print(w_0)
    print(w_1)
    hype_0 = get_hypothesis0(tf_data, w_0)
    hype_1 = get_hypothesis1(tf_data, w_1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(in_sample['x'], in_sample['y'], 'b+')
    ax.plot(hype_0['x'], hype_0['y'], 'r.')
    ax.plot(hype_1['x'], hype_1['y'], 'g.')
    plt.show()


