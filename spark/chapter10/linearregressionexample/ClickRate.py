# Practical Machine learning
# Linear Regression example
# Chapter 10

from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from math import log
from math import exp #  exp(-t) = e^-t
from operator import add


print "--------------------create context------------"
dacsample = "/home/dusan/Spark_Linear_Regression/dac_sample.txt"  # Should be some file on your system
sc = SparkContext("local[4]", "ClickRatePrediction") ##run on local with 4 cores, named it "ClickRatePrediction"
print "-------------------/create context------------"

print "--------------------parse text file-----------"
dacsample = open(dacsample)
dacData = [unicode(x.replace('\n', '').replace('\t', ',')) for x in dacsample]
print "-------------------/parse text file-----------"

print "-------------------RDD------------------------"
rawData  = (sc
            .parallelize(dacData, 4)  # Create an RDD
            .zipWithIndex()  # Enumerate lines
            .map(lambda (v, i): (i, v))  # Use line index as key
            .partitionBy(2, lambda i: not (i < 50026)) 
            .map(lambda (i, v): v))  # Remove index
print "-------------------/RDD------------------------"


print "-----split data to train, validation and test------"
weights = [.8, .1, .1] ## split to 80% train, 10% for validation and 10% for test 
seed = 42
# Use randomSplit with weights and seed
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit([.8 ,.1, .1], 42)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()

print "-----/split data to train, validation and test------"



def createOneHotDict(inputData):   
    return inputData.flatMap(lambda x : x).distinct().zipWithIndex().collectAsMap()



def parsePoint(point):
    
    """from each line remove first (label) and parse rest (features) into a list of (featureID, value) tuples.
    """
    y = map(lambda x : (x), point.split(","))
    y.pop(0)
    z1 = map(lambda x : (y.index(x), x), y)
    z2 =[(i,x) for i,x in enumerate(y)]
    return z2

parsedTrainFeat = rawTrainData.map(parsePoint) ##parse features



def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
    """create sparse vector of features
    """
    xx = filter(lambda item : ctrOHEDict.has_key(item) , rawFeats)
           
    sparseVector = SparseVector(numOHEFeats, sorted(map(lambda x : OHEDict[x], sorted(xx))), np.ones(len(xx)))
    
    return sparseVector


def parseOHEPoint(point, OHEDict, numOHEFeats):
    """create LabeledPoint in form of (label, sparse vector of features)
    """
    x = parsePoint(point)
    sparseVector = oneHotEncoding(x, OHEDict, numOHEFeats)
    
    return LabeledPoint(point[0], sparseVector)

ctrOHEDict = createOneHotDict(parsedTrainFeat) ## create sparse vector from features
numCtrOHEFeats = len(ctrOHEDict.keys()) ##number of features


OHETrainData = rawTrainData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats)) ##create train labeled points
OHETrainData.cache() ##cache

OHEValidationData = rawValidationData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats)) ##create validation labeled points
OHEValidationData.cache()

# running first model with fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True

print "-------------logistic regression with gradient descent---------"
model0 = LogisticRegressionWithSGD.train(data=OHETrainData, iterations=numIters, step=stepSize,regParam=regParam, regType=regType, intercept=includeIntercept) ##train model
sortedWeights = sorted(model0.weights)
print "------------/logistic regression with gradient descent---------"


def computeLogLoss(p, y):
   
    epsilon = 10e-12
    if (p==0):
      p = p + epsilon
    elif (p==1):
      p = p - epsilon
      
    if y == 1:
      z = -log(p)
    elif y == 0:
      z = -log(1 - p)
    return z




def getP(x, w, intercept):
    """ get probability
    """ 
    rawPrediction = 1 / (1 + exp( (- (w.T.dot(x.toArray())))  + intercept))
    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return rawPrediction



def evaluateResults(model, data):
    
    tp = data.map(lambda x: (getP(x.features, model.weights ,model.intercept), x.label))
    answer=tp.map(lambda x:computeLogLoss(x[0],x[1])).mean()
    return answer


classOneFracTrain = OHETrainData.map(lambda x : x.label).mean() ## mean of label (zeroR in weka)

logLossTrBase = OHETrainData.map(lambda x:computeLogLoss(classOneFracTrain, x.label)).mean() ##log loss of train data for mean of labels
LogLossTrLR0 = evaluateResults(model0, OHETrainData) ## evaluate model vs base


logLossValBase = OHEValidationData.map(lambda x:computeLogLoss(classOneFracTrain, x.label)).mean() ##log loss of validation data for mean of labels

print "----------------------find best hyperparameters for logistic regression--------"
## NOTE: maybe first hash features then iterate for best model
numIters = 500
regType = 'l2'
includeIntercept = True

## Initialize variables using values from initial model training
bestModel = None
bestLogLoss = 1e10

## iterate over two step sizes and two regParams
stepSizes = (1., 10.)
regParams = (1e-6, 1e-3)
for stepSize in stepSizes:
    for regParam in regParams:
        model = (LogisticRegressionWithSGD
                 .train(OHETrainData, numIters, stepSize, regParam=regParam, regType=regType,
                        intercept=includeIntercept))
        logLossVa = evaluateResults(model, OHEValidationData)
        print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
               .format(stepSize, regParam, logLossVa))
        if (logLossVa < bestLogLoss):
            bestModel = model
            bestLogLoss = logLossVa

print ('Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, bestLogLoss))

print "---------------------/find best hyperparameters for logistic regression--------"

print "-------------------predict on test data----------------------------------------"
OHETestData = rawTestData.map(lambda point: parseOHEPoint(point, ctrOHEDict, numCtrOHEFeats))

finalModel = LogisticRegressionWithSGD.train(OHETrainData, numIters, step=10., regParam=1e-3, regType=regType,intercept=includeIntercept)

labelsAndPredsTest = OHETestData.map(lambda lp: (lp.label, finalModel.predict(lp.features)))

print "number of observations in test data:"
print labelsAndPredsTest.count() ## 10014
print "number of true positives + true negatives:"
print labelsAndPredsTest.filter(lambda x: x[0] == x[1]).count() ## 7957


sc.stop() ##stop context


