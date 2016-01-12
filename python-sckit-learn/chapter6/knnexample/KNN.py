# Practical Machine learning
# k-Nearest neighbor example 
# Chapter 6

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors

class kNN():

    def __init__(self, k):
        self.k = k

    def _euclidian_distance(self, x1, x2):
        """Computes Euclidian Distance b/w two feature vectors
        X1 can be a numpy ndarray and x2 is numpy array
        """
        a= x1-x2
        a2 = a**2
        b = np.sum(a2, axis=1)
        c = np.sqrt(b)
        return c
#         return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X, y):
        """takes input of features and corresponding labels

        """
        self.X_data = X
        self.y = y

    def predict(self, X):
        """Classify features according to euclidian_distance from all data points

        Parameters:

        X:
        numpy ndarray

        """

        Xn = np.copy(X)

        preds = []
        # compute distance from all points
        for x1 in Xn:
            dist = self._euclidian_distance(self.X_data, x1)
            dist = np.vstack((dist, self.y)).T
            dist = dist[dist[:, 0].argsort(axis=0)][:,-1]
            # get a vote from top k
            pred = sts.mode(dist[0:self.k])[0][0]
            preds.append(pred)

        return np.array(preds)


# load dataset
data = pd.read_csv('data/iris.data', header=None)

h = .02  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# lets take only first two columns
X = data.iloc[:, :2].values
y = data.iloc[:, -1]

# convert to floats 0,1,2
y = y.apply(lambda x: 0 if x == 'Iris-setosa' else x)
y = y.apply(lambda x: 1 if x == 'Iris-versicolor' else x)
y = y.apply(lambda x: 2 if x == 'Iris-virginica' else x)

y = y.values

n_neighbors = 10

# ======================================
# my kNN
cl = kNN(n_neighbors)
cl.fit(X, y)


# ======================================
# scikit-learn
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)


# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Get predictions
knn_preds = clf.predict(np.c_[xx.ravel(), yy.ravel()])
scikit_preds = cl.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
knn_preds = knn_preds.reshape(xx.shape)
scikit_preds = scikit_preds.reshape(xx.shape)

#=====================================================
# plot kNN prediction
plt.figure()
plt.pcolormesh(xx, yy, knn_preds, cmap=cmap_light,
               vmin=knn_preds.min(), vmax=knn_preds.max())
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Prediction from kNN')
plt.savefig('knn_example.png')

#=====================================================
# plot scikit-learn predictions
plt.figure()
plt.pcolormesh(xx, yy, scikit_preds, cmap=cmap_light,
               vmin=scikit_preds.min(), vmax=scikit_preds.max())
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Prediction from scikit-learn')

plt.show()
