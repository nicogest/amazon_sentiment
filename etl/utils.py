
import numpy as np
import pandas as pd


def load(file_dir):

    file = pd.read_csv(file_dir, nrows=1000, error_bad_lines=False, header=None, names=['text'], sep='/n')

    return file


def split_xy(df):

    df['new_text'] = df['text'].apply(lambda x: x[11:])
    df['y'] = df['text'].apply(lambda x: int(x[9]) - 1)

    df_x = df['new_text']
    df_y = df['y']

    return df_x, df_y
