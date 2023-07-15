import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import scipy.io as sio
import sys
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, training_epochs=500, learning_rate=0.05, n_dim=2):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs  # this maybe too much?
        self.n_dim = n_dim
        self.n_output = 1
        self.batch_size = 40
        # self.W = tf.Variable(np.zeros((self.n_dim, self.n_output)), name="weight")
        # self.b = tf.Variable(np.zeros((1, self.n_output)), name="bias")

    def delete_para(self):
        # del self.learning_rate
        # del self.training_epochs
        self.W = None
        self.b = None
        self.training_epochs = None
        self.training_epochs = None
        self.n_dim = None
        self.n_output = None
        self.batch_size = None
        self.X = None
        self.Y = None
        self.pred = None
        self.init = None
        self.cost = None
        self.optimizer = None
        # del self.n_dim
        # del self.n_output
        # del self.batch_size

        return
        # print 'Destructors'

    def read_weights(self, weights=None, bias=None):
        if weights is not None:
            self.W = self.weight_initialization(False, weights)
        else:
            self.W = self.weight_initialization()
        if bias is not None:
            self.b = self.bias_initialization(False, bias)
        else:
            self.b = self.bias_initialization()

    def weight_initialization(self, initial_flag=True, values=None):
        if initial_flag == True:
            initial_value = np.random.randn(self.n_dim, self.n_output)
        else:
            initial_value = values

        weight = tf.Variable(initial_value, name="weight", dtype=tf.float64)
        return weight

    def bias_initialization(self, initial_flag=True, values=None):
        if initial_flag == True:
            initial_value = np.random.randn(1, self.n_output)
        else:
            initial_value = values
        bias = tf.Variable(initial_value, name="bias", dtype=tf.float64)
        return bias

    def linear_regression_model(self):
        # tf Graph Input
        self.X = tf.placeholder(tf.float64, [None, self.n_dim])  # CurrentObs
        self.Y = tf.placeholder(tf.float64, [None, self.n_output])  # Q_home and Q_away

        # Construct a linear model
        self.pred = self.X @ self.W + self.b

        # Mean squared error
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.pred - self.Y)))

        # Gradient descent
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def readout_linear_regression_model(self):
        # tf Graph Input
        self.X = tf.placeholder(tf.float64, [None, self.n_dim])  # CurrentObs

        # Construct a linear model
        self.pred = self.X @ self.W + self.b

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def compute_average_difference(self, listA, listB):
        diff = tf.abs(listA - listB)
        diff_all = tf.reduce_sum(diff) / tf.cast(tf.size(listA), tf.float64)
        max_diff = tf.reduce_max(diff)
        return diff_all, max_diff

    def gradient_descent(self, sess, train_X, train_Y):
        """
        Use tensorflow to do gradient descent
        :param train_X: training data (currentObs)
        :param train_Y: result value (q_values)
        :param n_samples: the number of instances
        :return: []
        """
        # use np.array to improve the old
        train_X = np.array(train_X)
        train_X = np.reshape(train_X, (-1, self.n_dim))
        train_Y = np.array(train_Y)

        sess.run(self.init)
        train_X_length = len(train_X)
        train_X_length_over_batch_size = train_X_length / self.batch_size
        train_Y_length = len(train_Y)

        random.sample(range(len(train_X)), len(train_X))

        # Fit all training data
        for epoch in range(self.training_epochs):
            random_number = np.random.permutation(len(train_X))
            train_X_reordered = train_X[random_number]
            train_Y_reordered = train_Y[random_number]

            if len(train_X) <= self.batch_size:
                cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: train_X, self.Y: train_Y})
                # print cost
            else:
                for i in range(0, int(train_X_length_over_batch_size)):
                    if i + 1 < train_X_length_over_batch_size:
                        input_, labels = train_X_reordered[
                                         i * self.batch_size:i * self.batch_size + self.batch_size], train_Y_reordered[
                                                                     i * self.batch_size:i * self.batch_size + self.batch_size]
                    else:
                        input_, labels = train_X_reordered[i * self.batch_size:], train_Y_reordered[
                                                                                  i * self.batch_size:]

                    cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: input_, self.Y: labels})


        trained_weights, trained_bias = sess.run([self.W, self.b])

        # pickle.dump(temp1, open('./temp_test_save/weights_1.p', 'w'))
        # pickle.dump(temp2, open('./temp_test_save/bias_1.p', 'w'))
        #
        temp, (average_diff, max_diff) = sess.run([self.pred, self.compute_average_difference(self.Y, self.pred)],
                                                feed_dict={self.X: train_X, self.Y: train_Y})

        print(sys.stderr, '(average_diff:{0}, max_diff:{1}, training_epochs:{2}, lr:{3})'.format(average_diff, max_diff, self.training_epochs, self.learning_rate))

        return trained_weights, trained_bias, average_diff


if __name__ == "__main__":
    # test_x = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    # test_y = [[2.0], [4.0], [6.0]]
    np.random.seed(0)

    # Generate input data
    n_samples = 100
    X1 = np.linspace(0, 2, n_samples)
    X2 = np.linspace(2, 4, n_samples)
    X3 = np.linspace(4, 6, n_samples)

    # Generate output data
    Y = 3 * X1 + 2 * X2 + 4 * X3 + np.random.randn(n_samples) * 0.33 + 5

    # Reshape to meet Scikit-learn input requirements
    X1 = np.array(X1).reshape((n_samples, 1))
    X2 = np.array(X2).reshape((n_samples, 1))
    X3 = np.array(X3).reshape((n_samples, 1))
    Y = np.array(Y).reshape((n_samples, 1))

    # Concatenate inputs
    test_x = np.concatenate((X1, X2, X3), axis=1)

    with tf.Session() as sess:
        # """read weights and bias"""
        # weight = pickle.load(open('./temp_test_save/weights_1.p', 'r'))
        # bias = pickle.load(open('./temp_test_save/bias_1.p', 'r'))
        # LR = LinearRegression(weights=weight, bias=bias)
        # LR.linear_regression_model()
        # temp = LR.gradient_descent(sess=sess, train_X=test_x, train_Y=test_y)
        # print temp

        """don't read weights and bias"""
        LR = LinearRegression(n_dim=len(test_x[0]))
        LR.read_weights()
        LR.linear_regression_model()
        temp = LR.gradient_descent(sess=sess, train_X=test_x, train_Y=Y)
        print(temp)

        reg = sklearn.linear_model.LinearRegression().fit(test_x, Y)
        y_pred = reg.predict(test_x)

        # The coefficients
        print('Coefficients: \n', reg.coef_)
        print('Mean squared error: %.2f'
              % mean_squared_error(Y, y_pred))
