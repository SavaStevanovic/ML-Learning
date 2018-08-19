import numpy as np


class Model():

    def __init__(self, learning_rate=0.01, iterations_count=50, shuffle_data=True, seed=1):
        self.learning_rate = learning_rate
        self.iterations_count = iterations_count
        self.seed = seed
        self.shuffle_data = shuffle_data
        self.w_initialized = False

    def fit(self, X, y):
        self.initialise_weights(X.shape[1])
        self.cost_ = []

        for _ in range(self.iterations_count):
            if(self.shuffle_data):
                X, y = self.shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
            self.cost_.append(np.average(cost))
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.initialise_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, yi in zip(X, y):
                self.update_weights(xi, yi)
        else:
            self.update_weights(X, y)
        return self

    def shuffle(self, X, y):
        permutation = self.rgen.permutation(len(y))
        return X[permutation], y[permutation]

    def initialise_weights(self, len):
        self.rgen = np.random.RandomState(self.seed)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + len)
        self.w_initialized = True

    def update_weights(self, xi, target):
        yi = self.activation(self.network_input(xi))
        error = target-yi
        self.w_[1:] += self.learning_rate*xi.dot(error)
        self.w_[0] += self.learning_rate*error
        cost = 0.5*error**2
        return cost

    def network_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.network_input(X)) >= 0.0, 1, -1)
