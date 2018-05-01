import tensorflow as tf
import numpy as np 
from .base-model import Model

class NeuralNet(Model):
    """
    Neural Network class for training networks with predefined settings
    """

    def __init__(self, n_input, n_hidden, dropout_keep_prob, lambda,
            num_epochs, batch_size, print_cost, plot_training, tf_seed)