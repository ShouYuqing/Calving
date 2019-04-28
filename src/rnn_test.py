"""
RNN prediction
"""
import os
import sys
import glob
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

# lib
import datagenerator
sys.path.append('../data/')
sys.path.append('../models/')

def test():
    # data generator
    data_dir = "../data/training_data"
    calv_num, files = datagenerator.file_name(data_dir)

    date_file_dir = "../data/calve_data.json"
    calv_dates = datagenerator.calv_date(calv_num = calv_num, file_dir = date_file_dir)

    activity = datagenerator.read_activity_data(calv_num = calv_num, calv_date = calv_dates, files = files, size = 14)  # (50, 14, 5)
    data, label = datagenerator.gene_data(num = len(calv_num), activity_data = activity)  # (50, 8, 7, 4) && (50, 8, 1)

    # validation data
    validate_input = data[40:50, :, :, :]
    validate_output = label[40:50, :, :]

    # model specification
    # parameters
    m = 14  # data length
    n = 4  # feature num
    len2 = 7  # length of window
    time_step = m - (len2 - 1)  # time_step size

    # model parameters
    lstm_size = 20
    lstm_layers = 2

    # placeholder
    x = tf.placeholder(tf.float32, [None, time_step, len2 * n], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, time_step], name='output_y')

    # cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size=lstm_size) for _ in range(lstm_layers)])

    # drop out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # drop = tf.contrib.rnn.DropWrapper(cell, output_keep_prob = keep_prob)

    # initial state
    initial_state = cell.zero_state(batch_size, tf.float32)

    # cell output
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    # output layer
    weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))
    b = tf.Variable(tf.ones([1]))
    bias = tf.zeros([1])
    outputs = tf.reshape(outputs, [-1, lstm_size])
    # logits = tf.sigmoid(tf.matmul(outputs, weights))
    logits = tf.matmul(outputs, weights) + b
    # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
    predictions = tf.reshape(logits, [-1, time_step])

    # cost
    cost = tf.losses.mean_squared_error(y_, predictions)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    # load and restore the model
    with tf.Session() as sess:
        saver.restore(sess, '../models/iter10001')


        #graph = tf.get_default_graph()
        #saver = tf.train.import_meta_graph('../models/iter10001.meta')
        #saver.restore(sess, '../models/iter10001')
        #input_x = graph.get_operation_by_name('input_x').outputs[0]
        #output_y = graph.get_operation_by_name('output_y').outputs[0]
        #keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]

    #x = gragh.get_tensor_by_name('Placeholder:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    #y = gragh.get_tensor_by_name('Placeholder_1:0')  # 获取输出变量
    #keep_prob = gragh.get_tensor_by_name('Placeholder_2:0')  # 获取dropout的保留参数
    #with tf.Session() as sess:


        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        #y = tf.get_collection('pred_network')[0]



        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        #input_x = graph.get_operation_by_name('input_x').outputs[0]
        #keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        # 使用y进行预测
        #sess.run(y, feed_dict={input_x:...., keep_prob:1.0})

def lstm_cell(lstm_size):
    """
    construct LSTM cell with size
    :param lstm_size: layer size
    :return: cell with size
    """
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

if __name__ == "__main__":
    test()