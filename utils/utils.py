import numpy as np 
import configparser
import pandas as pd
import os
import yaml
from shutil import copyfile

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

def load_config(path, save_path):
    with open(path, "r") as file:
        config = yaml.load(file)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    copyfile(path, save_path + "/config.yaml")
    return config

    