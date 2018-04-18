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


def hypothesis(data_set):
    data_set['z'] = 1
    m = np.asarray(data_set[['z', 'x']])
    m_trans = m.T
    m_pesudo = np.linalg.inv(m_trans.dot(m)).dot(m_trans)
    w = m_pesudo.dot(np.asarray(data_set[['y']]))
    weight = pd.DataFrame(w.T)
    weight.columns = ['w0', 'w1']
    return weight


def get_hypothesis(target_function_data, hypothesis_weight):
    w1 = hypothesis_weight['w1']
    x = target_function_data['x'].get_values()
    y = w1 * x
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

    result = np.zeros((1000, 3))

    for ii in range(1000):
        print(ii)
        # find bias
        tf_data = get_target_function(200)
        w_s = np.zeros((100, 2))
        for i in range(100):
            in_sample = get_target_function(2)
            w = hypothesis(in_sample)
            w_s[i] = w
        w_df = pd.DataFrame(w_s)
        w_df.columns = ['w0', 'w1']
        w_mean = w_df.mean()
        bias_vector = sample_error_vector(tf_data, w_mean)
        bias = square_mean_error(bias_vector)
        hypo_data = get_hypothesis(tf_data, w_mean)

        # find variance
        variance = np.zeros(100)
        for index, row in w_df.iterrows():
            v_vector = sample_error_vector(hypo_data, row)
            variance[index] = square_mean_error(v_vector)
        result[ii] = w_mean['w1'], bias, variance.mean()

    result_df = pd.DataFrame(result)
    result_df.columns = ['weight', 'bias', 'variance']
    print(result_df.mean())

    fig, ax = plt.subplots(1, 1)
    ax.plot(tf_data['x'], tf_data['y'], 'b.')
    ax.plot(hypo_data['x'], hypo_data['y'], 'r.')
    plt.show()


