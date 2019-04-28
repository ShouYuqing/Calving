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

def test(num_iteration):
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

    # prediction
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('../models/iter10001.meta')
        new_saver.restore(sess, '../models/iter10001')

        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        y = tf.get_collection('pred_network')[0]

        graph = tf.get_default_graph()

        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        # 使用y进行预测
        sess.run(y, feed_dict={input_x:...., keep_prob:1.0})
