"""
data generator for demo
"""
import numpy as np
import pandas as pd

def rand_f():


def demo_data(m = 30, n = 2, time_step = 15):
    """
    generate demo data with the size time_step*15*4
    :param m: data size
    :param n: data size
    :param time_step: time_step of RNN
    :return: data with the size of time_step*15*4
    """
    # sliding window to generate data
    data = np.zeros((m, n))
    for i in arange(n):
        data[0:m/2-1, i] = np.random.uniform(low=0.0, high=5.0, size=(int(m/2), 1)) + i
        data[m/2:m-1, i] = np.random.uniform()
    # generate positive
    # generate negative


def data_split():

