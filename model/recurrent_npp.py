
import numpy as np
import pandas as pd
import tensorflow as tf

class LSTM():

    def __init__(self, parameters):

        self.model = tf.keras.Sequential()
        self.parameters = parameters


    def model_creation(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.parameters['vocab_size'], self.parameters['embedding_dim'],
                                      input_length=self.parameters['max_length']),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss=self.parameters['loss'], optimizer=self.parameters['optimizer'],
                           metrics=self.parameters['metrics'])


    def fit(self, x, y, test_x, test_y):

        self.model_creation()
        self.model.fit(x, y, epochs=self.parameters['num_epochs'], validation_data=(test_x, test_y))



