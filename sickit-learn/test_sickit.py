from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import IFrame

# import load_iris function from datasets module
from sklearn.datasets import load_iris
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
       width=300, height=200)

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)

# print the iris data
print(iris.data)

# print the names of the four features
print(iris.feature_names)

# print integers representing the species of each observation
print(iris.target)

# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)

# check the types of the features and response
print(type(iris.data))
print(type(iris.target))

# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)

# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target

# TRAINING MODEL
# KNN

# print the shapes of X and y
print(X.shape)
print(y.shape)


knn = KNeighborsClassifier(n_neighbors=1)
print(knn)

knn.fit(X, y)
knn.predict([[3, 5, 4, 2]])
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)

# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
print(knn.predict(X_new))

# USING DIFFERENT MODEL
# LOGISTIC REGRESSION
# import the class

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
print(logreg.predict(X_new))
