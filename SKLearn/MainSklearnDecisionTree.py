from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from Ploting.Ploter import plot_decision_regions
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X_train, y_train)

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=model, test_idx=range(105, 150))
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# dot_data = export_graphviz(model, filled=True, rounded=True, class_names=[
#                            'Setosa', 'Versicolor', 'Virginica'], feature_names=['petal length', 'petal width'], out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('tree.png')