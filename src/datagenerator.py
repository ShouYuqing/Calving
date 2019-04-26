"""
data generator for demo
"""
import numpy as np
import pandas as pd

def gene_arr(length):
    """
    generate a array 0, 1, 2, ....., length - 1
    :param length: length of the array
    :return: the array
    """
    arr = np.zeros((length, 1))
    for i in np.arange(length):
        arr[i] = i
    return arr

def gene_rand(length):
    """
    generate a random array
    :param length: len
    :return: array
    """
    arr = np.zeros((length, 1))
    for i in np.arange(length):
        arr[i] = np.random.uniform(low = -5.0, high = 5.0)
    return arr

def demo_data(m = 30, n = 2, len = 15):
    """
    generate demo data with the size time_step*15*2 by sliding window over a fix data
    :param m: data size
    :param n: data size
    :param len: size of the window
    :return: data with the size of time_step*15*2
    """
    data = np.zeros((m, n))
    for i in np.arange(n):
        data[0:int(m/2), i] = (np.random.uniform(low = 0.0, high = 5.0, size = (int(m/2), 1)) + i).reshape(int(m/2))
        data[int(m/2):m, i] = (gene_arr(int(m/2)) + gene_rand(int(m/2))).reshape(int(m/2))
    # generate
    time_step = m - (len-1)
    dat = np.zeros((time_step, len, n))
    label = np.zeros((time_step, 1))
    for i in np.arange(time_step):
        if i == time_step - 1:
            label[i] = 1
        else:
            label[i] = 0
        for j in np.arange(n):
            dat[i, :, j] = data[i:i+len, j]
    return dat, label

def batch_data(batch_size, len = 15, m = 30):
    """
    generate batch for the data
    :param batch_size: size
    :return: batch of data
    """
    time_step = m - (len - 1)
    data = np.zeros((batch_size, time_step, len, 2))
    label = np.zeros((batch_size, time_step, 1))
    for i in np.arange(batch_size):
        d, l = demo_data()
        data[i, :, :, :] = d
        label[i, :, :] = l
    return data, label


