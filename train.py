from processors.preprocessor import Imputer, Normalizer
from processors.processor import Processor
import os
import pandas as pd
import numpy as np
import configparser
from pathlib import Path

Config = configparser.ConfigParser()
Config.read("config.ini")

print(Config["Imputer"]["strategy"])

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

X, Y = load_dataset(year=1, shuffle=True)

imputer = Imputer(strategy='mean')
normalizer = Normalizer(strategy="l2", norm_axis=1)
processor = Processor(10, 10, "regularize", 64)

X = imputer.fit_transform(abstracts=X)
X = normalizer.transform(abstracts=X)

x_train, y_train, x_dev, y_dev, x_test, y_test = processor.split_data(X, Y)
print(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape, y_test.shape)