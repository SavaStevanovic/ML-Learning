import numpy as np


class Model():

    w_: np.array
    errors_: np.array

    def __init__(self, learning_rate=0.01, iterations_count=50, seed=1):
        self.learning_rate = learning_rate
        self.iterations_count = iterations_count
        self.seed = seed

    def fit(self, X, y):
        rgen = np.random.RandomState(self.seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.iterations_count):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def network_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.network_input(X) >= 0.0, 1, -1)
