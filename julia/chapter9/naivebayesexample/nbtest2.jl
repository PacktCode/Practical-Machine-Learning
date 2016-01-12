# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

include("ntests1.jl")

# normal (variables on columns)
X = rand(40, 10)
ds = DataStats(10)
updatestats(ds, X[1:20, :])
updatestats(ds, X[21:end, :])

@assert all((cov(X) - cov(ds)) .< 0.0001)

# transposed (variables on rows)
X = rand(40, 10)
ds = DataStats(10, 2)
updatestats(ds, X')

@assert all((cov(X) - cov(ds)) .< 0.0001)

println("All OK")
