# Practical Machine learning
# Support Vector Machine example 
# Chapter 6

#To perform SVM on digits data we first load modules.
from sklearn import datasets
digits = datasets.load_digits()

# First we create an 'estimator'. This is a Python object which implements the fit, predict methods.
# This particular estimator implements Support Vector Classification.
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

# We now fit the training set data to the model (all data except last row)
SVC = clf.fit(digits.data[:-1], digits.target[:-1])

#and then test our model with the testing set data (last row)
# This gives us an integer representing which class it is in.
Test = clf.predict(digits.data[-1])

#print(digits.images[8])
#print(SVC)
print(Test)

#It is possible to save a model in the scikit by using Python’s built-in persistence model, namely pickle:
from sklearn import svm
clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data, iris.target
fit = clf.fit(X,y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
predict = clf2.predict(X[0])

print(predict)
#print(fit)
print(y[0])