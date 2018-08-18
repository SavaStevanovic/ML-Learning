import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron.Perceptron import Model as Perceptron
from Ploter import plot_decision_regions

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

pcp = Perceptron(iterations_count=10)
pcp.fit(X, y)
plt.plot(range(1, len(pcp.errors_)+1), pcp.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

plot_decision_regions(X, y, classifier=pcp)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
