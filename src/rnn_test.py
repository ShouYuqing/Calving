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
sys.path.append('../data/')
sys.path.append('../models/')
sys.path.append('../ext/')
import datagenerator
#import ssh_data


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
    batch_size = 20
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
        # validation
        val_x, val_y = datagenerator.gene_batch(batch_size=batch_size, data=validate_input, label=validate_output)
        result = sess.run(predictions, feed_dict={x: val_x.reshape(val_x.shape[0], val_x.shape[1], len2 * n),
                                                  y_: val_y.reshape([-1, time_step]), keep_prob: 1.0})
        print(result)

def lstm_cell(lstm_size):
    """
    construct LSTM cell with size
    :param lstm_size: layer size
    :return: cell with size
    """
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

if __name__ == "__main__":
    test()