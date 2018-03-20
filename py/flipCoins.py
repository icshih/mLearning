import scipy.stats as ss
import numpy as np
import pandas as pd


def flipCoins(num_coins, num_flip):
    sample = np.zeros((num_coins, num_flip))
    for i in range(num_coins):
        sample[i] = ss.binom.rvs(1, 0.5, size=num_flip)
    return pd.DataFrame(sample)


def find_small(dataset):
    coin = 10
    heads = 10
    for i, f in enumerate(dataset):
        num_head = dataset.iloc[i].sum()
        if num_head < heads:
            coin = i
            heads = num_head
    return coin, heads


def run_simulation(num_run):
    first = list()
    rand = list()
    mini = list()
    for i in range(num_run):
        output = flipCoins(1000, 10)
        c_first = output.iloc[0]
        c_rand = output.sample(n=1)
        c_min = find_small(output)
        first.append(c_first.sum())
        rand.append(c_rand.values.sum())
        mini.append(c_min[1])
    data = {'first': first, 'random': rand, 'mini': mini}
    return pd.DataFrame(data)


if __name__ == '__main__':
    data = run_simulation(100000)
    c_first_mean = data['first'].mean()
    c_rand_mean = data['random'].mean()
    c_min_mean = data['mini'].mean()
    print(c_first_mean, c_rand_mean, c_min_mean)