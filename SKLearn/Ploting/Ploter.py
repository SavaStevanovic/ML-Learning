import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
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
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


def iris_model_testing(model):
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)
    print('Labels counts in y:', np.bincount(y))
    print('Labels counts in y_train:', np.bincount(y_train))
    print('Labels counts in y_test:', np.bincount(y_test))

    standardScaler = StandardScaler()
    standardScaler.fit(X_train)

    X_train_std = standardScaler.transform(X_train)
    X_test_std = standardScaler.transform(X_test)

    model.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=model, test_idx=range(105, 150))
    plt.xlabel('sepal length [normalized]')
    plt.ylabel('petal length [normalized]')
    plt.legend(loc='upper left')
    plt.show()
