import tensorflow as tf
import numpy as np 
from base_model import Model

class NeuralNet(Model):
    """
    Neural Network class for training networks with predefined settings
    """

    def __init__(self, layers, dropout_prob, alpha, lmbda,
                num_epochs, batch_size, init_type, iter_per_cost, plot):
        """
        Initiates all config variables and sets them.
        """

        # Set all variables
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.lmbda = lmbda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_type = init_type
        self.iter_per_cost = iter_per_cost
        self.plot = plot

    def initialize_params(self, abstracts, labels):
        """
        Initializes all weights and biases
        Args:
            Abstracts: dict of training data including the X and Y
            Labels: Number of data classifications
        Output:
            Weights: Dictionary of all weights
            Biases: Dictionary of all biases
        """

        self.weights = dict()
        self.biases = dict()
        if self.init_type == "Xavier":
            self.initializer = tf.contrib.layers.xavier_initializer()

        for i in range(len(self.layers)):
            num_input = self.layers[i]
            num_output = len(labels) if i == len(self.layers - 1) else self.layers[i + 1] 
            name = "input" if i == 0 else "output" if i == len(self.layers) else str(i + 1)
            w_shape = [num_input, num_output]
            b_shape = [num_output]

            self.weights["W_" + name] = tf.Variable(self.initializer(w_shape), name = "W_{}".format(name))
            self.biases["b_" + name] = tf.Variable(tf.zeros(b_shape), name = "b_{}".format(name))

        self.X_input = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
        self.Y_input = tf.placeholder(tf.int32, shape=(None))
    
    def train(self, abstracts, labels):
        """
        """
        X = abstracts["X"]
        Y = abstracts["Y"]

        # do the batch stuff here
            


    
        
            


    



