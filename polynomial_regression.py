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
        self.params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}

        # Gradients
        self.grads = {key: np.zeros_like(self.params[key]) for key in self.params}

        # Intermediate parameters needed for backprop
        self.cache = {}

    def zero_grad(self):
        """ Zero out the gradients.
        """
        self.grads = {key: np.zeros_like(self.params[key]) for key in self.params}

    def forward(self, X):
        """ Forward pass.

        Args:
            X: numpy array of shape (n_samples, 1)

        Returns:
            y_pred: numpy array of shape (n_samples, 1)
        """
        # Hidden layer
        z1 = np.dot(self.params['W1'], X) + self.params['b1']
        a1 = np.tanh(z1)

        # Output layer
        z2 = np.dot(self.params['W2'], a1) + self.params['b2']
        y_pred = z2

        # Store intermediate results for backprop
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2

        return y_pred

    def backward(self, y_true):
        """ Backward pass.

        Args:
            y_true: numpy array of shape (n_samples, 1)
        """
        # Output layer gradients
        dz2 = 2 * (self.cache['z2'] - y_true)
        self.grads['W2'] = np.dot(dz2, self.cache['a1'].T)
        self.grads['b2'] = np.sum(dz2, axis=1, keepdims=True)

        # Hidden layer gradients
        dz1 = np.dot(self.params['W2'].T, dz2) * (1 - self.cache['a1']**2)
        self.grads['W1'] = np.dot(dz1, self.cache['X'].T)
        self.grads['b1'] = np.sum(dz1, axis=1, keepdims=True)

    def update_params(self, learning_rate=0.01):
        """ Update the parameters.

        Args:
            learning_rate: learning rate
        """
        self.params = {key: self.params[key] - learning_rate * self.grads[key] for key in self.params}

    def update_params_momentum(self, learning_rate=0.01, momentum=0.9):
        """ Update the parameters with momentum.

        Args:
            learning_rate: learning rate
            momentum: momentum
        """
        if not hasattr(self, 'v'):
            self.v = {k: self.grads[k] for k, v in self.grads.items()}

        self.v = {k: momentum * self.v[k] + (1 - momentum) * self.grads[k] for k, v in self.v.items()}
        self.params = {key: self.params[key] - learning_rate * self.v[key] for key in self.params}

    @property
    def num_parameters(self):
        """ Return the number of parameters.
        """
        return sum(np.prod(v.shape) for v in self.params.values())

    def fit(self, X, y, n_epochs=100, learning_rate=0.01,
            batch_size=1, verbose=False, desc='polynomial_regression'):
        """ Fit the model.

        Args:
            X: numpy array of shape (n_samples, 1)
            y: numpy array of shape (n_samples, 1)
            n_epochs: number of epochs
            learning_rate: learning rate
            batch_size: batch size
            verbose: verbosity
        """
        # print info
        if verbose:
            print('>>>>>>>>>>>>')
            print('Training the model with following parameters:')
            print(' - description: {}'.format(desc))
            print(' - num of hidden layers: {}'.format(self.n_hidden))
            print(' - num of epochs: {}'.format(n_epochs))
            print(' - learning rate: {}'.format(learning_rate))
            print(' - batch size: {}'.format(batch_size))
            print(' - num of trainable parameters: {}'.format(self.num_parameters))
            print('<<<<<<<<<<<<')

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
                self.backward(y_batch)

                # Update parameters
                self.update_params_momentum(learning_rate)

            # Set progress bar description
            pbar.set_description('Epoch %3d: loss = %.6f' % (epoch + 1, loss))

def train_model(X, y, desc='polynomial_regression'):
    """ Train the model.
    """
    n_samples = X.shape[0]
    # train model
    model = PolynomialRegression(n_hidden=15, n_output=1)
    model.fit(X, y, n_epochs=100, 
                    learning_rate=0.001, 
                    batch_size=5, 
                    verbose=True, 
                    desc=desc)
    
    # plot model
    plt.figure(figsize=(30, 16))
    plt.suptitle(desc, fontsize=20)
    x_plot = np.linspace(-1.0, 1.0, n_samples)
    y_plot = f(x_plot)
    y_pred = [model.forward(x.reshape(-1, 1)) for x in x_plot]
    y_pred = np.array(y_pred).reshape(-1, 1)
    plt.plot(x_plot, y_plot, 'b-', linewidth=5)
    plt.plot(x_plot, y_pred, 'r-', linewidth=5)
    plt.plot(X, y, 'ko')
    plt.legend(['true model', 'predicted model', 'noisy data samples'], fontsize=25)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid()
    plt.savefig(f'{desc}.png')
    plt.show()

if __name__ == '__main__':
    # set random seed
    np.random.seed(25)
    n_samples = 2500

    # generate data
    coeffs = [1.0, 7.5, -5.0, -1.0, 4.5, -7.0]
    # polynomial function
    f = lambda x: sum(coeffs[i] * x**i for i in range(len(coeffs)))
    X, y = generate_data(f, noise_std=1.0, n_samples=n_samples)
    train_model(X, y, desc='polynomial_regression')