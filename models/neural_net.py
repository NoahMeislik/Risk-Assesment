import tensorflow as tf
import numpy as np 
from models.base_model import Model
from processors.processor import Processor

class NeuralNet(Model):
    """
    Neural Network class for training networks with predefined settings
    """

    def __init__(self, layers, dropout_prob, alpha, lmbda, num_epochs, init_type, optimizer_type,
                 dev_percent, test_percent, batch_type, batch_size, shuffle, iter_per_cost, plot, 
                 save_path):
        """
        Initiates all config variables and sets them.
        """

        # NN Parameter Vars
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.lmbda = lmbda
        self.num_epochs = num_epochs
        self.init_type = init_type
        self.optimizer_type = optimizer_type

        # Processor
        self.dev_percent = dev_percent
        self.test_percent = test_percent
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Progress
        self.iter_per_cost = iter_per_cost
        self.plot = plot
        self.save_path = save_path
        

    def initialize_params(self, X, Y, labels):
        """
        Initializes all weights and biases
        Args:
            Abstracts: dict of training data including the X and Y
            Labels: What the data can be classified as Ex: cats, dogs, birds and sheep a list
        Output:
            All initialized variables
        """
        self.X = X
        self.Y = Y

        self.processor = Processor(self.X, self.Y, self.dev_percent, self.test_percent, self.batch_type, self.batch_size, self.shuffle)

        self.train_set_x, self.train_set_y, dev_set_x, dev_set_y, test_set_x, test_set_y = self.processor.split_data()

        self.processor = Processor(self.train_set_x, self.train_set_y, self.dev_percent, self.test_percent, self.batch_type, self.batch_size, self.shuffle)

        self.X_input = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
        self.Y_input = tf.placeholder(tf.float32, shape=(None))
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        if self.init_type == 0:
            self.initializer = tf.contrib.layers.xavier_initializer()

        self.neurons = self.X_input
        for i in range(len(self.layers)):   
            num_input = self.layers[i]
            num_output = 1 if i == len(self.layers) - 1 else self.layers[i + 1] 
            name = "input" if i == 0 else "output" if i == len(self.layers) else str(i + 1)
            w_shape = [num_input, num_output]
            b_shape = [num_output]

            self.W = tf.Variable(self.initializer(w_shape), dtype=tf.float32, name = "W_{}".format(name))
            self.b = tf.Variable(tf.zeros(b_shape, dtype=tf.float32), name = "b_{}".format(name))

            self.neurons = tf.add(tf.matmul(self.neurons, self.W), self.b)
            
            if i < len(self.layers) - 1:
                self.neurons = tf.nn.dropout(self.neurons, keep_prob=self.dropout_keep_prob)
                self.neurons = tf.nn.relu(self.neurons)

        self.logits = self.neurons
        # To-Do add regularization here
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.Y_input))

        if self.optimizer_type == 0:
            optimizer = tf.train.AdamOptimizer(learning_rate = self.alpha)
            # To-Do add more optimizers

        self.train_op = optimizer.minimize(self.loss_op)

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
    
    def train(self):
        """
        To be run after running init_params. Takes the values created in init_params and trains them with the data.
        """

        with tf.Session() as sess:

            sess.run(self.init)

            for epoch in range(self.num_epochs):

                avg_cost = 0
                num_batches = int(len(self.Y)/self.batch_size)
                for _ in range(num_batches):
                    batch_x, batch_y = self.processor.next_batch()
                    

                    _, c = sess.run([self.train_op, self.loss_op], feed_dict={self.X_input: batch_x, self.Y_input: batch_y, self.dropout_keep_prob: self.dropout_prob})
                    
                    avg_cost = c / num_batches

                print("The average cost of epoch " + str(epoch+1) + " is: " + str(avg_cost))


    
        
            


    



