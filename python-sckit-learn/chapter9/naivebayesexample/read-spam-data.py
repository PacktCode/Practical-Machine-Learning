# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

from datatypes import Dataset
from classifier import naive_bayes, svm, naive_bayes_custom, knn
from feature_selection import univariate_feature_selection, lda, pca

from sklearn.cross_validation import train_test_split
from numpy import mean, var, sum, diag, shape

def load_spam_ds():
    """
    Loads the data from file and build the dataset in scikit format.

    () -> Dataset
    """

    data = []
    target = []
    i = 0
    with open("data/spambase.data", "r") as f:
        for line in f:
            # Removes \r\n from line
            line = line.replace("\n","").replace("\r","")
            
            items = line.split(",")
            features = [float(item) for item in items[:-1]]
            spam_class = int(items[-1])
            data.append(features)
            target.append(spam_class)
    
    return Dataset(data, target)

def split_train_test(ds):
    """
    Given the dataset, split in two datasets:
    One is the Training set. Other is the Test set.
    The proportion is 80% to 20% Respectively
    
    Dataset -> Dataset, Dataset
    """

    samples_train, samples_test, classes_train, classes_test = train_test_split(ds.data, ds.target, test_size=0.2)
    training_set = Dataset(samples_train, classes_train)
    test_set = Dataset(samples_test, classes_test)
    return training_set, test_set

def run(n=0, dimension_reduction=univariate_feature_selection, learning=naive_bayes_custom):
    """
    Starts the classification Pipeline
    """
    ds = load_spam_ds()
    if n > 0 and n < len(ds.data):
        ds = dimension_reduction(ds, n)
    evaluate(ds, learning)

def evaluate(ds, classifier_class, iterations=10):
    ''' 
    Train a given classifier n times
    and prints his confusion matrix and the accuracy of the classifier
    with a margin of error (by Chebychev Inequation)
    '''
    results = []
    for i in range(iterations):
        training_set, test_set = split_train_test(ds)
        classifier = classifier_class(training_set)
        cm = 1.0 * classifier.classify(test_set) / len(test_set.data)
        results += [cm]
    cm_mean = mean(results, axis=0)
    cm_variance = var(results, axis=0)
    
    print ("Accuracy of", sum(diag(cm_mean))*100, "% (+-", iterations * sum(diag(cm_variance)), ") with", (1 - 1.0/(iterations*iterations)),  "of certain." )
    print ("\nConfusion Matrix:\n",cm_mean,"\n")
    
if __name__ == "__main__":
    algo=[naive_bayes_custom, naive_bayes, knn, svm]
    feature=[univariate_feature_selection, pca, lda]
    num=[1,10,0]
    for n in num:
        for f in feature:
            if (n==0):
                print("\nUsing all features")
            else:
                print("\nUsing",n,"feature(s) (", f.__name__, ")" )
            print("=======================================================\n")    
            for a in algo:
                print("* Learning Algorithm:", a.__name__)
                run(n, f, a)
            if (n==0):
                break
