from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from Ploting.Ploter import plot_decision_regions
import matplotlib.pyplot as plt


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

model = Perceptron(max_iter=1000, eta0=0.01, random_state=1)
model.fit(X_train_std, y_train)

y_pred = model.predict(X_test_std)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=model, test_idx=range(105, 150))
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

print('Misclassified samples: %d ' % (y_test != y_pred).sum())
print('Accuracy: %.2f ' % accuracy_score(y_test, y_pred))
