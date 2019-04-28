"""
RNN for calving time prediction
"""
import os
import glob
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

# lib
import datagenerator
sys.path.append('../data/')


def train(iterations, load_iter, batch_size = 30):
    """
    RNN for calving time prediction
    :param iterations: training iteration
    :param load_iter: continue training from checkpoint
    :param batch_size: batch_size
    """
    # data generator
    data_dir = "../data/training_data"
    calv_num, files = file_name(data_dir)

    date_file_dir = "../data/calve_data.json"
    calv_dates = calv_date(calv_num=calv_num, file_dir=date_file_dir)

    activity = read_activity_data(calv_num=calv_num, calv_date=calv_dates, files=files, size=14)  # (50, 14, 5)

    data, label = gene_data(num=len(calv_num), activity_data=activity)# (50, 8, 7, 4) && (50, 8, 1)

    train_input = data[0:40, :, :, :]
    train_output = label[0:40, :, :]

    validate_input =
    validate_output =

    # parameters
    m = 14 # total length of data for each cow
    n = 4 # each date's feature
    len = 7 # length of sliding window
    time_step = m - (len - 1) # time_step for training

    # model parameters
    lstm_size = 20
    lstm_layers = 2

    # placeholder
    x = tf.placeholder(tf.float32, [None, time_step, len*n], name = 'input_x')
    y_ = tf.placeholder(tf.float32, [None, time_step], name = 'output_y')

    # cell
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size = lstm_size) for _ in range(lstm_layers)])

    # drop out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #drop = tf.contrib.rnn.DropWrapper(cell, output_keep_prob = keep_prob)

    # initial state
    initial_state = cell.zero_state(batch_size, tf.float32)

    # cell output
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    # output layer
    weights = tf.Variable(tf.truncated_normal([lstm_size, 1], stddev=0.01))
    bias = tf.zeros([1])
    outputs = tf.reshape(outputs, [-1, lstm_size])
    logits = tf.sigmoid(tf.matmul(outputs, weights))
    # [batch_size*binary_dim, 1] ==> [batch_size, binary_dim]
    predictions = tf.reshape(logits, [-1, time_step])

    # cost
    cost = tf.losses.mean_squared_error(y_, predictions)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # train
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        iteration = 1
        for i in range(iterations):
            # read data
            input_x, input_y = datagenerator.batch_data(batch_size = batch_size)
            _, loss = sess.run([optimizer, cost], feed_dict={x: input_x.reshape(input_x.shape[0], input_x.shape[1], len*n), y_: input_y.reshape([-1, time_step]), keep_prob: 0.5})

            if iteration % 100 == 0:
                print('Iter:{}, Loss:{}'.format(iteration, loss))
            iteration += 1

        # validation
        val_x, val_y = datagenerator.batch_data(batch_size=batch_size)
        result = sess.run(predictions, feed_dict={x: val_x.reshape(val_x.shape[0], val_x.shape[1], len*n), y_: val_y.reshape([-1, time_step]), keep_prob: 1.0})

        print(result)

    # save model

    # print loss

def lstm_cell(lstm_size):
    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--load_iters", type=int,
                        dest="load_iter", default=0,
                        help="load iters")
    parser.add_argument("--iters", type=int,
                        dest="iterations", default=1000,
                        help="number of iterations")
    args = parser.parse_args()
    train(**vars(args))