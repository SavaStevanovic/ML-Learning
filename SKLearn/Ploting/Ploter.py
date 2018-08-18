import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x0_min, x0_max = X[:, 0].min()-1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min()-1, X[:, 1].max() + 1

    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, resolution),
                           np.arange(x1_min, x1_max, resolution))
    Z = classifier.predict(np.array([xx0.ravel(), xx1.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx0, xx1, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0],
                    y=X[y == c1, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=c1,
                    edgecolors='black')
