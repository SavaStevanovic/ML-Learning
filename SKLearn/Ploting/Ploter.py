import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from DimensionReduction.FeatureSelection.SequentialBackwardSelection import SBS


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
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()


def wine_model_testing(model):
    df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    print('Class labels', np.unique(df_wine['Class label']))
    df_wine.head()
    X = df_wine.iloc[:, 1:].values
    y = df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model.fit(X_train_std, y_train)
    print("Trainiong accuracy:", model.score(X_train_std, y_train))
    print("Test accuracy:", model.score(X_test_std, y_test))

    sbs = SBS(model, k_features=1)
    sbs.fit(X_train_std, y_train)
    k_features = [len(k) for k in sbs.subsets_]
    plt.plot(k_features, sbs.scores_, marker='o')
    plt.ylim(0.7, 1.02)
    plt.xlabel('Accuracy')
    plt.ylabel('Number of features')
    plt.grid()
    plt.show()
    k_best = np.argmax(list(reversed(sbs.scores_)))
    best_features_indeces = list(list(reversed(sbs.subsets_))[k_best])
    print(df_wine.columns[1:][best_features_indeces])

    model.fit(X_train_std[:, best_features_indeces], y_train)
    print("Trainiong accuracy:", model.score(X_train_std[:, best_features_indeces], y_train))
    print("Test accuracy:", model.score(X_test_std[:, best_features_indeces], y_test))
