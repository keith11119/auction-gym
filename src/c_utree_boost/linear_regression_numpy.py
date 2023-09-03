import random

import numpy as np
import pickle
import scipy.io as sio
import sys
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, training_epochs=50, learning_rate=0.05, n_dim=2, batch_size=200):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs  # this maybe too much?
        self.n_dim = n_dim
        self.n_output = 1
        self.batch_size = batch_size

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
            self.W = weights
        else:
            self.W = np.random.randn(self.n_dim, self.n_output)
        if bias is not None:
            self.b = bias
        else:
            self.b = np.random.randn(1)


    def cost_function(self, X, y):
        predictions = self.predict(X)
        cost = np.mean((predictions - y) ** 2)
        return cost

    def predict(self, X):
        predictions = X @ self.W + self.b
        return predictions

    def gradient_descent(self, train_X, train_Y):
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

        train_X_length = len(train_X)
        train_X_length_over_batch_size = train_X_length / self.batch_size
        train_Y_length = len(train_Y)

        random.sample(range(len(train_X)), len(train_X))

        # Fit all training data
        for epoch in range(self.training_epochs):
            random_number = np.random.permutation(train_X_length)
            train_X_reordered = train_X[random_number]
            train_Y_reordered = train_Y[random_number]

            if len(train_X) <= self.batch_size:
                input_, labels = train_X_reordered, train_Y_reordered
                cost = self.cost_function(input_, labels)

                predictions = self.predict(input_)
                diff = predictions - labels
                grad_W = (1.0 / len(input_)) * (input_.T @ diff)
                grad_b = (1.0 / len(input_)) * np.sum(diff)

                # Not using adam optimizer but normal gradient descent
                self.W = self.W - self.learning_rate * grad_W
                self.b = self.b - self.learning_rate * grad_b

            else:
                for i in range(0, int(train_X_length_over_batch_size)):
                    if i + 1 < train_X_length_over_batch_size:
                        input_, labels = train_X_reordered[
                                         i * self.batch_size:i * self.batch_size + self.batch_size], train_Y_reordered[
                                                                     i * self.batch_size:i * self.batch_size + self.batch_size]
                    else:
                        input_, labels = train_X_reordered[i * self.batch_size:], train_Y_reordered[
                                                                                  i * self.batch_size:]

                    #cost, _ = sess.run([self.cost, self.optimizer], feed_dict={self.X: input_, self.Y: labels})
                    cost = self.cost_function(input_, labels)

                    predictions = self.predict(input_)
                    diff = predictions - labels
                    grad_W = (1.0 / len(input_)) * (input_.T @ diff)
                    grad_b = (1.0 / len(input_)) * np.sum(diff)

                    # Not using adam optimizer but normal gradient descent
                    self.W = self.W - self.learning_rate * grad_W
                    self.b = self.b - self.learning_rate * grad_b

        avg_diff = np.mean(np.abs(self.predict(train_X_reordered) - train_Y_reordered))
        max_diff = np.max(np.abs(self.predict(train_X_reordered) - train_Y_reordered))
        print(f'(average_diff:{avg_diff}, max_diff:{max_diff}, training_epochs:{self.training_epochs}, lr:{self.learning_rate})')

        return self.W , self.b, avg_diff


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


    """don't read weights and bias"""
    LR = LinearRegression(n_dim=len(test_x[0]))
    LR.read_weights()
    temp = LR.gradient_descent(train_X=test_x, train_Y=Y)
    print(temp)

    reg = sklearn.linear_model.LinearRegression().fit(test_x, Y)
    y_pred = reg.predict(test_x)

    # The coefficients
    print('Coefficients: \n', reg.coef_)
    print('Mean squared error: %.2f'
          % mean_squared_error(Y, y_pred))
