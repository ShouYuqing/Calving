"""
newest version
LSTM training
"""
import os
import sys
import glob
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

# lib
sys.path.append('../data/')
sys.path.append('../ext/')
import datagenerator
import ssh_data


def train(iterations, load_iter, batch_size = 20):
    """
    RNN for calving time prediction
    :param iterations: training iteration
    :param load_iter: continue training from checkpoint
    :param batch_size: batch_size
    """
    # update data
    #ssh_data.ssh_get(src="-r /home/cloud/date/training_data")

    # data generator
    data_dir = "../data/training_data"
    calv_num, files = datagenerator.file_name(data_dir)

    date_file_dir = "../data/calve_data.json"
    calv_dates = datagenerator.calv_date(calv_num=calv_num, file_dir=date_file_dir)

    activity = datagenerator.read_activity_data(calv_num=calv_num, calv_date=calv_dates, files=files, size=12)  # (50, 12, 5)

    data, label = datagenerator.gene_data(num= len(calv_num), activity_data=activity)# (50, 8, 5, 4) && (50, 8, 1)

    # split training and testing
    train_input = data[0:40, :, :, :]
    train_output = label[0:40, :, :]

    validate_input = data[40:50, :, :, :]
    validate_output = label[40:50, :, :]

    # parameters
    m = 12 # data length
    n = 4 # feature num
    len2 = 5 # length of window
    #time_step = m - (len2 - 1) # time_step size
    time_step = 1

    # model parameters
    lstm_size = 20
    lstm_layers = 2

    # placeholder
    x = tf.placeholder(tf.float32, [None, time_step, len2*n], name = 'input_x')
    y_ = tf.placeholder(tf.float32, [None, time_step], name = 'output_y')

    # cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size = lstm_size) for _ in range(lstm_layers)])

    # drop out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #drop = tf.contrib.rnn.DropWrapper(cell, output_keep_prob = keep_prob)

    # initial state
    initial_state = cell.zero_state(batch_size, tf.float32)

    # cell output
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state = initial_state)

    # output layer
    weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))
    b = tf.Variable(tf.ones([1]))
    bias = tf.zeros([1])
    outputs = tf.reshape(outputs, [-1, lstm_size])
    logits = tf.sigmoid(tf.matmul(outputs, weights))
    #logits = tf.matmul(outputs, weights) + b
    # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
    predictions = tf.reshape(logits, [-1, time_step])

    # cost
    cost = tf.losses.mean_squared_error(y_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

    # train
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        iteration = 1
        for i in range(iterations):
            # read data
            input_x, input_y = datagenerator.gene_batch(batch_size = batch_size, data = train_input, label = train_output)#(8, 8, 5, 4)--->(8, 5, 4)
            for j in np.arange(8):
                _, loss = sess.run([optimizer, cost], feed_dict={x: input_x[:, j, :, :].reshape((input_x.shape[0], 1, input_x.shape[2]*input_x.shape[3])), y_: input_y[:, j].reshape((input_y.shape[0], 1)), keep_prob: 0.5})
            if iteration % 100 == 0:
                print('Iter:{}, Loss:{}'.format(iteration, loss))
            iteration += 1

        # save model
        saver = tf.train.Saver()
        saver.save(sess, "../models/iter" + str(iteration))

        # validation
        val_x, val_y = datagenerator.gene_batch(batch_size = batch_size, data = validate_input, label = validate_output)
        result = sess.run(predictions, feed_dict={x: val_x[:, 7, :, :].reshape((val_x.shape[0], 1, val_x.shape[2]*val_x.shape[3])), y_: val_y[:, 7].reshape((val_y.shape[0], 1)), keep_prob: 1.0})
        cost = sess.run(cost, feed_dict={x: val_x[:, 3, :, :].reshape((val_x.shape[0], 1, val_x.shape[2]*val_x.shape[3])), y_: val_y[:, 3].reshape((val_y.shape[0], 1)), keep_prob: 1.0})
        print(val_y[:, 3].reshape((val_y.shape[0], 1)))
        print(result)
        print(cost)

def lstm_cell(lstm_size):
    """
    construct LSTM cell with size
    :param lstm_size: layer size
    :return: cell with size
    """
    return tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias = 1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_iters", type=int,
                        dest="load_iter", default=0,
                        help="load iters")
    parser.add_argument("--iters", type=int,
                        dest="iterations", default=10000,
                        help="number of iterations")
    args = parser.parse_args()
    train(**vars(args))