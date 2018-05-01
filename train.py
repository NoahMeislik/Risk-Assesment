from processors.preprocessor import Imputer, Normalizer
import os
import pandas as pd
import numpy as np


def load_dataset(year, shuffle=False):
    """Loads chosen data set, mixes it and returns."""
    file_path = "C:/Users/noahm/Desktop/risk-assesment/data/" + "{}year.csv".format(year)
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

imputer.fit_transform(abstracts=X)
normalizer.transform(abstracts=X)

