from sklearn.neighbors import KNeighborsClassifier
from Ploting.Ploter import wine_model_testing

model = KNeighborsClassifier(
    n_neighbors=5, p=2, metric='minkowski')
wine_model_testing(model)
