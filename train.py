import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_train, Y_train)
