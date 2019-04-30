"""
data generator
"""
import json
import numpy as np
import os
import datetime

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

"""
Data cleansing
    raw data:
        calve_data.json
        training_data
    steps: 
        read id data into file --> read calving date through id --> read back data before the calving date
    
    file:     training_data             calve_data                      training_data
    Through sliding window to generate data
    return: time-series data directly used for RNN training and validating
"""

def file_name(file_dir):
    """
    search the calves that has data
    :param file_dir: file dir
    :return: list of calv_num(string)
    """
    for root, dirs, files in os.walk(file_dir):
        print("total data:")
        print(files)
    calv_num = []
    for file in files:
        # split dates "XXXX-XX-XX"
        calv_num.append(os.path.splitext(file)[0])
    return calv_num, files

def calv_date(calv_num, file_dir):
    """
    read calving date
    :param calv_num: calv_num list
    :param file_dir: json file dir
    :return: list of calving date
    """
    # dict
    calv_dates = {}
    f = open(file_dir, encoding='utf-8')
    read_data = json.load(f)
    for num in calv_num:
        calv_dates[str(num)] = read_data[num]
    return calv_dates

def getdate(date, days):
    """
    Use lib datatime to deal with string time processing
    return list of date before n days
    :param date: present date
    :param days: prior days
    :return: list of date
    """
    # list
    date_list = []
    date_s= date.split('-')
    the_date = datetime.datetime(int(date_s[0]), int(date_s[1]), int(date_s[2]))
    # construct dates
    j = 1
    for i in np.arange(days):
        result_date = the_date + datetime.timedelta(days = -j)
        d = result_date.strftime('%Y-%m-%d')
        date_list.append(d)
        j = j + 1
    return date_list


def read_activity_data(calv_num, calv_date, files, size, data_dir = "../data/training_data/"):
    """
    generate cow activity data for window sliding
    :param calv_num: list of id
    :param calv_date: date(a single date)
    :param files: list of "id.json"
    :param size: days
    :param data_dir: training data file
    :return: activity data prior n days before calving (cow_num x data length x feature size)
    """
    # acivity data
    activity = np.zeros((len(calv_num), size, 5))
    for i in np.arange(len(calv_num)):
        #print("--------start reading cow " + str(calv_num[i]) + "--------")
        # read all dates
        calv_dates = getdate(calv_date[str(calv_num[i])], days=size)
        # read .json data of each cow
        file_dir = data_dir + files[i]
        f = open(file_dir, encoding='utf-8')
        read_data = json.load(f)# all the activity data for a single cow
        m = size - 1
        for j in calv_dates:
            activity[i, m, :] = read_data[j]
            m = m - 1
    return activity

def gene_data(num, activity_data, len = 5):
    """
    generate training data
    sliding window to generate data&label
    feature selection
    :param num: cow's number
    :param activity_data: all the data
    :param len: window size
    :return: data, label
    """
    time_step = activity_data.shape[1] - (len - 1)
    data = np.zeros((num, time_step, len, 4))
    label = np.zeros((num, time_step, 1))
    for n in np.arange(activity_data.shape[0]):
        for i in np.arange(time_step):
            label[n, i] = time_step - i
            #if i == time_step - 1:
            #    label[n, i] = 1
            #else:
            #    label[n, i] = 0
            m = 0
            # select features
            for j in np.array([0, 1, 2, 4]):
                data[n, i, :, m] = activity_data[n, i:i + len, j]
                m = m + 1
    return data, label

def gene_pred(data_dir = "../data/predict_data/", latest_date = "2019-03-19", size = 12, num_feature = 5):
    """
    generate data used for prediction
    :param data_dir: prediction date dir
    :param latest_date: latest date
    :param size: data length(days of calving date)
    :param num_feature: number of feature
    :return: array of id and data
    """
    calv_num, files = file_name(data_dir)
    dates = getdate(date = latest_date, days = size)
    pred_data = np.zeros((len(calv_num), size, num_feature))
    id = np.zeros((len(calv_num), 1))
    for i in np.arange(len(calv_num)):
        # array of id
        id[i] = calv_num[i]
        # read .json data of each cow
        file_dir = data_dir + files[i]
        f = open(file_dir, encoding='utf-8')
        read_data = json.load(f)# all the activity data for a single cow
        m = 0
        for j in dates:
            print(dates)
            pred_data[i, m, :] = read_data[j]
            m = m + 1
    return pred_data, id

def gene_batch(batch_size, data, label):
    """
    generate batch from data, label
    :param batch_size: size
    :param data: data
    :param label: label
    :return: batch
    """
    time_step = data.shape[1]
    num_feature = data.shape[3]
    len = data.shape[2]
    batch_data = np.zeros((batch_size, time_step, len, num_feature))
    batch_label = np.zeros((batch_size, time_step, 1))
    for i in np.arange(batch_size):
        n = int(np.random.randint(0, data.shape[0] - 1))
        batch_data[i, :, :, :] = data[n, :, :, :]
        batch_label[i, :, :] = label[n, :, :]
    return batch_data, batch_label

if __name__ == "__main__":
    # test file_name()
    data_dir = "../data/training_data"
    calv_num, files = file_name(data_dir)

    # test calv_date()
    date_file_dir = "../data/calve_data.json"
    calv_dates = calv_date(calv_num = calv_num, file_dir = date_file_dir)

    # test getdate()
    #dates = getdate(calv_dates[calv_num[1]], days=5)

    # test read_activity_data()
    activity = read_activity_data(calv_num = calv_num, calv_date = calv_dates, files = files, size = 12)# (50, 12, 5)

    # test gene_data()
    data, label = gene_data(num = len(calv_num), activity_data = activity)

    # test gene_batch()
    input, output = gene_batch(batch_size = 20, data = data, label = label)

    # test gene_pred()
    p_data, id = gene_pred(data_dir = "../data/predict_data1/")
    print(p_data.shape)
    pre_data, _ = gene_data(num = p_data.shape[0], activity_data = p_data)


