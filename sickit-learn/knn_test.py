from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# import some data to play with
iris = datasets.load_iris()

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'],iris['target'])
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=6,p=2,weights='uniform')

iris['data'].shape
prediction=knn.predict(X_new)
X_new.shape
print('prediction {}'.format(prediction))