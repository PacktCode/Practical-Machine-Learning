# Practical Machine learning
# Support Vector Machines example
# Chapter 6

# Example: ExImage Recognition with Support Vector Machines

import sklearn as sk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# print 'IPython version:', IPython.__version__
# print 'numpy version:', np.__version__
# print 'scikit-learn version:', sk.__version__
# print 'matplotlib version:', matplotlib.__version__
from sklearn.datasets import fetch_olivetti_faces

# fetch the faces data
faces = fetch_olivetti_faces()

# print faces.DESCR

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
		
		print_faces(faces.images, faces.target, 20)
		print_faces(faces.images, faces.target, 400)

# Build training and testing sets
from sklearn.svm import SVC
svc_1 = SVC(kernel='linear')
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        faces.data, faces.target, test_size=0.25, random_state=0)
		
# Perform 5-fold cross-validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print ("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), sem(scores))
		
	evaluate_cross_validation(svc_1, X_train, y_train, 5)
	
# measure precision and recall on the evaluation set, for each class.
train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)