"""
RNN test
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

def test(num_iteration, ):
    with tf.Session() as sess:
        # read model
        saver = tf.train.import_meta_graph(str(num_iteration) + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint('../models/'))
