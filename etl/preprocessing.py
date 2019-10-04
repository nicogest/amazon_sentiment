
import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess(df_x, df_y, vocab_size, embedding_dim, max_lenght, trunc_type, oov_tok):

    df_x = df_x.values
    df_y = df_y.values

