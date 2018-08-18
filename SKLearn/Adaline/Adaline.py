import numpy as np

class Model():

    def __init__(self, learning_rate=0.01, iterations_count=50, seed=1):
        self.learning_rate = learning_rate
        self.iterations_count = iterations_count
        self.seed = seed

    def fit(self, X, y):
        rgen = np.random.RandomState(self.seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.iterations_count):
            net_input = self.network_input(X)
            net_output=self.activation(net_input)
            errors = y - net_output
            self.w_[1:] += self.learning_rate*X.T.dot(errors)
            self.w_[0] += self.learning_rate*sum(errors)
            cost=(errors**2).sum()/2
            self.cost_.append(cost);     
        return self

    def network_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self,X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.network_input(X)) >= 0.0, 1, -1)
