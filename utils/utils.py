import numpy as np 
import configparser
import pandas as np
import os

def load_dataset(year, shuffle=False):
    """Loads chosen data set, mixes it and returns."""
    file_path = "data/" + "{}year.csv".format(year)
    df = pd.read_csv(file_path, na_values='?')
    Y = df['class'].values
    X = df.drop('class', axis=1).values
    if shuffle:
        shuffled_idx = np.random.permutation(len(Y))
        X = X[shuffled_idx, :]
        Y = Y[shuffled_idx]
    return X, Y