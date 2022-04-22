""" This file contains functions and demonstrations
    of a toy example of a regression problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import tqdm

def generate_data(f, n_samples=100, noise_std=0.1):
    """ Generate data for a regression problem.

    Args:
        f: function to generate data for.
        n_samples: number of samples to generate.
        noise_std: standard deviation of noise.

    Returns:
        X: numpy array of shape (n_samples, 1)
        y: numpy array of shape (n_samples, 1)
    """
    X = np.random.uniform(-1.0, 1.0, (n_samples, 1))
    y = f(X) + np.random.normal(0, noise_std, (n_samples, 1))
    return X, y

class L2Loss(object):
    """ L2 loss function.
    """
    def __call__(self, y_pred, y_true):
        """ Compute the loss.

        Args:
            y_pred: numpy array of shape (n_samples, 1)
            y_true: numpy array of shape (n_samples, 1)

        Returns:
            loss: float
        """
        loss = np.mean((y_pred - y_true)**2)
        return loss

class PolynomialRegression(object):
    """ Polynomial regression model.
    """
    def __init__(self, n_hidden, n_output):
        """ Initialize the MLP.

        Args:
            n_hidden: number of hidden units
            n_output: number of output units
        """
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Initialize weights
        self.W1 = np.random.normal(0, 0.1, (self.n_hidden, 1))
        self.b1 = np.zeros((self.n_hidden, 1))
        self.W2 = np.random.normal(0, 0.1, (self.n_output, self.n_hidden))
        self.b2 = np.zeros((self.n_output, 1))

        # parameters for backprop
        self.cache = {}

    def forward(self, X):
        """ Forward pass.

        Args:
            X: numpy array of shape (n_samples, 1)

        Returns:
            y_pred: numpy array of shape (n_samples, 1)
        """
        # Hidden layer
        z1 = np.dot(self.W1, X) + self.b1
        a1 = np.tanh(z1)

        # Output layer
        z2 = np.dot(self.W2, a1) + self.b2
        y_pred = z2

        # store intermediate results for backprop
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2

        return y_pred

    def backward(self, y_true):
        """ Backward pass.

        Args:
            y_true: numpy array of shape (n_samples, 1)

        Returns:
            grads: dictionary of gradients
        """
        # Initialize gradients
        grads = {}
        grads['dW1'] = np.zeros_like(self.W1)
        grads['db1'] = np.zeros_like(self.b1)
        grads['dW2'] = np.zeros_like(self.W2)
        grads['db2'] = np.zeros_like(self.b2)

        # Output layer gradients
        dz2 = 2 * (self.cache['z2'] - y_true)
        grads['dW2'] = np.dot(dz2, self.cache['a1'].T)
        grads['db2'] = np.sum(dz2, axis=1, keepdims=True)

        # Hidden layer gradients
        dz1 = np.dot(self.W2.T, dz2) * (1 - self.cache['a1']**2)
        grads['dW1'] = np.dot(dz1, self.cache['X'].T)
        grads['db1'] = np.sum(dz1, axis=1, keepdims=True)

        return grads

    def update_params(self, grads, learning_rate=0.01):
        """ Update the parameters.

        Args:
            grads: dictionary of gradients
            learning_rate: learning rate
        """
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

    def update_params_momentum(self, grads, learning_rate=0.01, momentum=0.9):
        """ Update the parameters with momentum.

        Args:
            grads: dictionary of gradients
            learning_rate: learning rate
            momentum: momentum
        """
        if not hasattr(self, 'v'):
            self.v = {}
            self.v = {k: grads[k] for k, v in grads.items()}

        self.v['dW1'] = momentum * self.v['dW1'] + (1 - momentum) * grads['dW1']
        self.v['db1'] = momentum * self.v['db1'] + (1 - momentum) * grads['db1']
        self.v['dW2'] = momentum * self.v['dW2'] + (1 - momentum) * grads['dW2']
        self.v['db2'] = momentum * self.v['db2'] + (1 - momentum) * grads['db2']

        self.W1 -= learning_rate * self.v['dW1']
        self.b1 -= learning_rate * self.v['db1']
        self.W2 -= learning_rate * self.v['dW2']
        self.b2 -= learning_rate * self.v['db2']

    def fit(self, X, y, n_epochs=100, learning_rate=0.01,
            batch_size=1, verbose=False):
        """ Fit the model.

        Args:
            X: numpy array of shape (n_samples, 1)
            y: numpy array of shape (n_samples, 1)
            n_epochs: number of epochs
            learning_rate: learning rate
            batch_size: batch size
            verbose: verbosity
        """
        # Initialize loss
        loss = 0.0

        # Initialize number of batches
        n_batches = int(np.ceil(X.shape[0] / batch_size))

        # Loop over epochs
        pbar = tqdm.tqdm(range(n_epochs))
        for epoch in pbar:
            for i in range(n_batches):
                # Get batch
                batch_start = i * batch_size
                batch_end = batch_start + batch_size

                # Get batch data
                X_batch = X[batch_start:batch_end].T
                y_batch = y[batch_start:batch_end].T

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = L2Loss()(y_pred, y_batch)

                # Backward pass
                grads = self.backward(y_batch)

                # Update parameters
                self.update_params_momentum(grads, learning_rate)

            # Print loss
            if verbose:
                pbar.set_description('Epoch %3d: loss = %.6f' % (epoch + 1, loss))

def train_model():
    """ Train the model.
    """
    # generate data
    n_samples = 1000
    coeffs = [1.0, 7.5, -5.0, -1.0, 4.5, -7.0]
    # polynomial function
    f = lambda x: sum(coeffs[i] * x**i for i in range(len(coeffs)))
    X, y = generate_data(f, noise_std=0.5, n_samples=n_samples)
    
    # train model
    model = PolynomialRegression(n_hidden=10, n_output=1)
    model.fit(X, y, n_epochs=50, learning_rate=0.007, batch_size=2, verbose=True)
    
    # plot model
    plt.figure(figsize=(20, 12))
    x_plot = np.linspace(-1.0, 1.0, n_samples)
    y_plot = f(x_plot)
    y_pred = [model.forward(x.reshape(-1, 1)) for x in x_plot]
    y_pred = np.array(y_pred).reshape(-1, 1)
    plt.plot(x_plot, y_plot, 'b-')
    plt.plot(x_plot, y_pred, 'r-')
    plt.plot(X, y, 'ko')
    plt.legend(['true model', 'predicted model', 'noisy data samples'])
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid()
    plt.savefig('polynomial_regression.png')
    plt.show()

if __name__ == '__main__':
    # set random seed
    np.random.seed(25)
    train_model()