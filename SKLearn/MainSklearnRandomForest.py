from sklearn.ensemble import RandomForestClassifier
from Ploting.Ploter import iris_model_testing


model = RandomForestClassifier(
    criterion='gini', n_estimators=25, random_state=1, n_jobs=12)
iris_model_testing(model)
