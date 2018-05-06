import numpy as np 
import sklearn

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
        

