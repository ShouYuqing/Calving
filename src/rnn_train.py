"""
tensorflow implementation of RNN
"""
import os
import glob
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

# lib
import datagenerator
sys.path.append('../ext/')

def train(iterations, load_iter, batch_size = 1):
    # read data

    # model

    # train

    # print loss


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