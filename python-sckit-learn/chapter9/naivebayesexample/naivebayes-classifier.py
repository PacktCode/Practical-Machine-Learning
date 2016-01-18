# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from numpy import mean, std, tile, sqrt, power, sum, log

class naive_bayes():
    def __init__(self, training_set):
        """
            Builds the classifier given the Dataset.
            Dataset is splited using cross-validation with 80-20 proportion.

            ds (Dataset) -> GaussianNB
        """
        self.classifier = GaussianNB()
        self.classifier.fit(training_set.data, training_set.target)

    def classify(self, test_set):

        """
            Given my classifier and test_set,
            Evaluate my score and give me a Confusion Matrix.

            classifier (GaussianNB), test (Dataset) -> ConfusionMatrix
        """
        predictions = self.classifier.predict(test_set.data)

        return confusion_matrix(test_set.target, predictions)

class naive_bayes_custom():
    def __init__(self, training_set):
        '''
        In this implementation, the training of the Naive Bayes Classifier will be
        is just calculate the ratio of ocurrencies of each class and 
        also the mean and standard deviation of the training set
        for each feature by classes.
        '''
        spam_set = training_set.data[training_set.target==1]
        not_spam_set = training_set.data[training_set.target==0]

        self.spam_mean = mean(spam_set, axis=0)
        self.not_spam_mean = mean(not_spam_set, axis=0)

        self.spam_std = std(spam_set, axis=0)
        self.not_spam_std = std(not_spam_set, axis=0)

        self.spam_prob = 1.0*len(spam_set)/len(training_set.data)
    
    def __priorProb(self, features, mean, std):
        '''
            An auxiliary function used on classify method.
            It gives the result of the log of pdf of the Normal Distribution 
            (for a given mean and standard deviation)
            on a features' vector
 
        ''' 
        mean = tile( mean, ( len(features), 1 ) )
        std = tile( std, ( len(features), 1 ) )

        #Get rid of the division by zero error =)
        std[std==0] = 0.00000000001

        constant = sqrt(2*(3.1415))
        factor = - log( constant * std )
        expoent = - power( features - mean, 2) / (2 * power(std,2) ) 
        
        return factor + expoent 

    def classify(self, test_set):
        spam_prior = self.__priorProb(test_set.data, self.spam_mean, self.spam_std)
        spam_prob = sum( spam_prior, axis=1 ) + log( self.spam_prob )

        not_spam_prior = self.__priorProb(test_set.data, self.not_spam_mean, self.not_spam_std) 
        not_spam_prob = sum( not_spam_prior, axis=1 ) + log(1.0 - self.spam_prob)
        
        predictions = (spam_prob > not_spam_prob)*1
        

        return confusion_matrix(test_set.target, predictions)

class svm():
    '''
        Just wraps sklearn svm classifier
    '''

    def __init__(self, training_set):
        self.classifier = SVC(kernel='rbf')
        self.classifier.fit(training_set.data, training_set.target)
        

    def classify(self, test_set):
        predictions = self.classifier.predict(test_set.data)
        return confusion_matrix(test_set.target, predictions)

class knn():

    def __init__(self, training_set):
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(training_set.data, training_set.target)

    def classify(self, test_set):
        predictions = self.classifier.predict(test_set.data)
        return confusion_matrix(predictions, test_set.target)
