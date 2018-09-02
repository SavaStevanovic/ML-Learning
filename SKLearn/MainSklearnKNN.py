from sklearn.neighbors import KNeighborsClassifier
from Ploting.Ploter import iris_model_testing


model = KNeighborsClassifier(
    n_neighbors=5,p=2,metric='minkowski')
iris_model_testing(model)
