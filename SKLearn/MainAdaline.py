import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Adaline.Adaline import Model as Adaline
from Ploting.Ploter import plot_decision_regions

def PlotModel(ax,learning_rate):
    model=Adaline(iterations_count=10,learning_rate=learning_rate)
    model.fit(X,y)
    ax.plot(range(1, len(model.cost_)+1), np.log(model.cost_), marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(Sum-squared-error)')
    ax.set_title("Adaline - Learning rate %s" % learning_rate)

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[:100, [0, 2]].values

fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))

learning_rate=0.01
PlotModel(ax[0],learning_rate)

learning_rate=0.0001
PlotModel(ax[1],learning_rate)
plt.show()


