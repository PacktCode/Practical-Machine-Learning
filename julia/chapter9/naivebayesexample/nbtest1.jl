# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

using StatsBase
using NaiveBayes

function test_multinomial()
    print("testing MultinomialNB... ")
    m = MultinomialNB([:a, :b, :c], 5)
    X = [1 2 5 2;
         5 3 -2 1;
         0 2 1 11;
         6 -1 3 3;
         5 7 7 1]
    y = [:a, :b, :a, :c]
    
    fit(m, X, y)
    @assert predict(m, X) == y
    println("OK")
end

function test_gaussian()
    print("testing GaussianNB... ")
    n_obs = 100
    m = GaussianNB([:a, :b, :c], 5)
    X = randn(5, n_obs)
    y = sample([:a, :b, :c], n_obs)
    
    fit(m, X, y)
    accuracy = sum(predict(m, X) .== y) / n_obs
    println(accuracy)
    println("OK")
end


