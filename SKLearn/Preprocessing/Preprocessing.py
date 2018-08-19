import numpy as np


def normalize(X):
    return (X-X.mean(axis=0))/X.std(axis=0)
