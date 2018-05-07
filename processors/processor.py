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
    def __init__(self, X, Y, dev_percent, test_percent, batch_type, batch_size, shuffle):
        
        self.X = X
        self.Y = Y

        self.dev_percent = dev_percent
        self.test_percent = test_percent
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(self.Y)

    def split_data(self):
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


        dev_samples = math.floor((self.dev_percent / 100) * len(self.Y))
        test_samples = math.floor((self.test_percent / 100) * len(self.Y))
        train_samples = len(self.Y) - (dev_samples + test_samples)

        train_set_x = self.X[:train_samples, :]
        train_set_y = self.Y[:train_samples]
        dev_set_x = self.X[train_samples:dev_samples + train_samples, :]
        dev_set_y = self.Y[train_samples:dev_samples + train_samples]
        test_set_x = self.X[dev_samples + train_samples:, :]
        test_set_y = self.Y[dev_samples + train_samples:]

        return train_set_x, train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y

    
    def next_batch(self):
        """
        Returns the next batch in the dataset based off the number of epochs and batch size

        Reuqired Args:
            none

        Output:
            next_batch: the next batch from the dataset

        """
        # To-Do implement batch gd
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            if self.shuffle:
                permutation = numpy.arrange(self._num_examples)
                numpy.random.shuffle(permutation)
                self.X = self.X[permutation, :]
                self.Y = self.Y[permutation]

            start = 0
            self._index_in_epoch = self.batch_size

            assert(self.batch_size <= self._num_examples)
        end = self._index_in_epoch
        return self.X[start:end, :], self.Y[start:end]



            
        

