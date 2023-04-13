import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

import mlflow

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

knn.score(X_test, y_test)
