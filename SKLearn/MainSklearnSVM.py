from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

model = SVC(kernel='rbf', C=5., random_state=1, gamma=1)
model.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=model, test_idx=range(105, 150))
plt.xlabel('sepal length [normalized]')
plt.ylabel('petal length [normalized]')
plt.legend(loc='upper left')
plt.show()
