
import yaml
import argparse
import os
import tensorflow as tf

import numpy as np
import pandas as pd

from etl import utils, preprocessing
from model import recurrent_npp

# config import
confi_dir = "/Users/debenetti nicolo/PycharmProjects/NLP/amazon_sentiment/config/"
config = yaml.safe_load(open(confi_dir+"config_main.yaml", 'r'))

# load train and test
train = utils.load(config['input']['folder']+config['input']['train'])
test = utils.load(config['input']['folder']+config['input']['test'])

# split in x and y
train_x, train_y = utils.split_xy(train)
test_x, test_y = utils.split_xy(test)

# preprocess data
train_x, train_y, \
 test_x, test_y = preprocessing.preprocess(train_x, train_y, test_x, test_y, config['preprocessing']['vocab_size'],
                                        config['preprocessing']['embedding_dim'], config['preprocessing']['max_length'],
                                        config['preprocessing']['trunc_type'], config['preprocessing']['oov_tokQ'])

# model running
lstm = recurrent_npp.LSTM(parameters=config['model_lstm'])
lstm.fit(train_x, train_y, test_x, test_y)

print('done')