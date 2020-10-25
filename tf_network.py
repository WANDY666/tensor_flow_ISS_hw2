import tensorflow as tf
from tensorflow.contrib.layers import flatten


class neuralNetwork:

    def __init__(self):
        '''
        Define some basic parameters here
        '''

        pass

    def Net(self, input):
        '''
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        '''
        conv1 = tf.layers.conv2d(  # shape (28, 28, 1)
            inputs=input,
            filters=16,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )           # (28, 28, 16)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=2,
            strides=2
        )           # (14, 14, 16)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )           # (14, 14, 32)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=2,
            strides=2
        )           # (7, 7, 32)
        flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
        return tf.layers.dense(flat,10)

    def forward(self, input):
        '''
        Forward the network
        '''
        return self.Net(input)

    def init_weight(self, shape):
        '''
        Init weight parameter.
        '''
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    def init_bias(self, shape):
        '''
        Init bias parameter.
        '''
        b = tf.zeros(shape)
        return tf.Variable(b)
