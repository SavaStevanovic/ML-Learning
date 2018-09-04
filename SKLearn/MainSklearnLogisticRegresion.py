from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Ploting.Ploter import plot_decision_regions
from Ploting.Ploter import wine_model_testing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


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

weights, params = [], []

for c in np.arange(-5, 5):
    model = LogisticRegression(C=10.**c, random_state=1)
    model.fit(X_train_std, y_train)
    weights.append(model.coef_[1])
    params.append(100.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.xlabel('weight coefficient')
plt.ylabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=model,
                      test_idx=range(int(X.shape[0]*0.7), X.shape[0]))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

model = LogisticRegression(penalty='l1', C=1.)
wine_model_testing(model)
