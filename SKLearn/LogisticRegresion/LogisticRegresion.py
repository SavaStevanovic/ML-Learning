import numpy as np


class Model():

    def __init__(self, learning_rate=0.01, iterations_count=50, seed=1):
        self.learning_rate = learning_rate
        self.iterations_count = iterations_count
        self.seed = seed

    def fit(self, X, y):
        self.initialise_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.iterations_count):
            self.cost_.append(self.update_weights(X, y))
        return self

    def shuffle(self, X, y):
        permutation = self.rgen.permutation(len(y))
        return X[permutation], y[permutation]

    def initialise_weights(self, len):
        self.rgen = np.random.RandomState(self.seed)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + len)

    def update_weights(self, X, target):
        output = self.activation(self.network_input(X))
        error = target-output
        self.w_[1:] += self.learning_rate*X.T.dot(error)
        self.w_[0] += self.learning_rate*error.sum()
        cost = -target.dot(np.log(output))-(1-target).dot(np.log(1-output))
        return cost

    def network_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, z):
        return 1./(1.+np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.network_input(X)) >= 0.5, 1, -1)
