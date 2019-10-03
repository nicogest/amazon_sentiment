# utils

import numpy as np
import pandas as pd

def load_csv(folder, file):

    df = pd.read_csv(folder+file)

    return df

