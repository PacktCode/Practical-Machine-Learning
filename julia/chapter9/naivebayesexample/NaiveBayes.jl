# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9
module NaiveBayes

export NBModel,
       MultinomialNB,
       GaussianNB,
       fit,
       predict,
       predict_proba

include("nbtypes.jl")
include("core.jl")

end
