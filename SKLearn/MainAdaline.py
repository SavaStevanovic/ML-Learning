import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Adaline.Adaline import Model as Adaline
from Ploting.Ploter import plot_decision_regions
from Preprocessing.Preprocessing import normalize


def MultyPlotModel(ax, learning_rate, X, y):
    model = Adaline(iterations_count=10, learning_rate=learning_rate)
    model.fit(X, y)
    #model.partial_fit(X, y)
    ax.plot(range(1, len(model.cost_)+1), np.log(model.cost_), marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.set_title("Adaline - Learning rate %s" % learning_rate)


def PlotModel(learning_rate, X, y, iterations_count):
    model = Adaline(iterations_count=iterations_count,
                    learning_rate=learning_rate)
    model.fit(X, y)

    plot_decision_regions(X, y, model)
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.title("Adaline - Learning rate %s" % learning_rate)
    plt.show()

    plt.plot(range(1, len(model.cost_)+1), model.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[:100, [0, 2]].values

X_norm = normalize(X)
learning_rate = 0.01
PlotModel(learning_rate, X_norm, y, 15)
