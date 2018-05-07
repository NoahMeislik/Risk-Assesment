import numpy as np 
import sklearn
import math

class Processor():
    """
    Required Args:
        dev_percent: Percentage of data that will become the dev set
        test_percent: Percentage of data that will become the test set        
        batch_type: This can be of any amount I want including batch, mini batch, stochastic, etc...
        batch_size: Size of batches
        
    Output:
        Depending on batch type it will output split data in batches
    """
    def __init__(self, dev_percent, test_percent, batch_type, batch_size):
        
        self.dev_percent = dev_percent
        self.test_percent = test_percent
        self.batch_type = batch_type
        self.batch_size = batch_size

    def split_data(self, X, Y):
        """
        Splits data into three parts: train, dev, test

        Required Args:
            X: Array of X data
            Y: Array of Y data

        Output:
            train_x
            train_y
            dev_x
            dev_y
            test_x
            text_y
        """


        dev_samples = math.floor((self.dev_percent / 100) * len(Y))
        test_samples = math.floor((self.test_percent / 100) * len(Y))
        train_samples = len(Y) - (dev_samples + test_samples)

        train_set_x = X[:train_samples, :]
        train_set_y = Y[:train_samples]
        dev_set_x = X[train_samples:dev_samples + train_samples, :]
        dev_set_y = Y[train_samples:dev_samples + train_samples]
        test_set_x = X[dev_samples + train_samples:, :]
        test_set_y = Y[dev_samples + train_samples:]

        return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y


