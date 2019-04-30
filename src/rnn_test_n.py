"""
RNN prediction
"""
import os
import sys
import glob
import json
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

# lib
sys.path.append('../data/')
sys.path.append('../models/')
sys.path.append('../ext/')
import datagenerator
import ssh_data


def test(id):
    """
    RNN prediction
    :param id: data file id
    :return:
    """
    # update data
    #ssh_get(src = "-r /home/cloud/predict_data" + str(id))

    # data generator
    p_data, id = datagenerator.gene_pred(data_dir = "../data/predict_data" + str(id) + '/', latest_date = "2019-03-19", size = 12, num_feature = 5)

    # model specification
    # parameters
    m = 12  # data length
    n = 4  # feature num
    len2 = 5  # length of window
    time_step = 1  # time_step size

    # model parameters
    batch_size = p_data.shape[0]
    lstm_size = 20
    lstm_layers = 2

    # placeholder
    x = tf.placeholder(tf.float32, [None, time_step, len2 * n], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, time_step], name='output_y')

    # cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size=lstm_size) for _ in range(lstm_layers)])

    # drop out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #drop = tf.contrib.rnn.DropWrapper(cell, output_keep_prob = keep_prob)

    # initial state
    initial_state = cell.zero_state(batch_size, tf.float32)

    # cell output
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    # output layer
    weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))
    b = tf.Variable(tf.ones([1]))

    outputs = tf.reshape(outputs, [-1, lstm_size])
    logits = tf.sigmoid(tf.matmul(outputs, weights))
    #logits = tf.matmul(outputs, weights) + b
    # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
    predictions = tf.reshape(logits, [-1, time_step])

    # cost
    cost = tf.losses.mean_squared_error(y_, predictions)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    saver = tf.train.Saver()

    # data for front-end
    save_result = np.zeros((batch_size, 1))
    # load and restore the model
    with tf.Session() as sess:
        saver.restore(sess, '../models/iter10001')
        # validation
        #val_x, val_y = datagenerator.gene_batch(batch_size=batch_size, data=validate_input, label=validate_output)
        val_x, _ = datagenerator.gene_data(num=p_data.shape[0], activity_data=p_data, len = len2)
        val_x = np.array(val_x)
        result = sess.run(predictions, feed_dict={x: val_x[:, 7, :, :].reshape((val_x.shape[0], 1, val_x.shape[2] * val_x.shape[3])), keep_prob: 1.0})# all result from ../prediction_data(num, 8, 5, 4)
        print(result)
        save_result[:] = result

    # result as dict
    predict_result = {}
    for i in np.arange(id.shape[0]):
        predict_result[str(int(id[i]))] = save_result[i, 0]
    # result into json
    file_dir = '../data/predict_result.json'
    # nums = {"name": "Mike", "age": 12}
    with open(file_dir, 'w') as file_obj:
        print("---------write into json---------")
        json.dump(predict_result, file_obj)

    # send result to the front-end
    ssh_data.ssh_send()
    print("--------send to the front end---------")


def lstm_cell(lstm_size):
    """
    construct LSTM cell with size
    :param lstm_size: layer size
    :return: cell with size
    """
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_id", type=int,
                        dest="id", default=1,
                        help="data file id: 1/2")

    args = parser.parse_args()
    test(**vars(args))