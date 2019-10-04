
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(train_x, train_y, test_x, test_y,  vocab_size, embedding_dim, max_length, trunc_type, oov_tok):

    # conversion from a pandas object to a numpy array
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values

    # create a Tokenizer object
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # fit the Tokenizer on the train_x --> sentences
    tokenizer.fit_on_texts(train_x)
    # create the word index --> to check if this is substantial vs to check
    word_index = tokenizer.word_index
    # create a sequence of words after the training of the Tokenizer
    sequences = tokenizer.texts_to_sequences(train_x)
    # create a padded object for training
    train_x = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    # create a tokenized version of the test
    testing_sequences = tokenizer.texts_to_sequences(test_x)
    # create a padded version of the test
    test_x = pad_sequences(testing_sequences, maxlen=max_length)

    return train_x, train_y, test_x, test_y

