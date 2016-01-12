# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

using Distributions

include("datastats.jl")

# Base type for Naive Bayes models.
# Inherited classes should have at least following fields:
#  c_counts::Dict{C, Int64} - count of ocurrences of each class
#  n_obs::Int64             - total number of observations
abstract NBModel{C}

#####################################
#####  Multinomial Naive Bayes  #####
#####################################

type MultinomialNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    x_counts::Dict{C, Vector{Number}}  # count/sum of occurrences of each var
    x_totals::Vector{Number}           # total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


# Multinomial Naive Bayes classifier
#
# classes : array of objects
#     Class names
# n_vars : Int64
#     Number of variables in observations
# alpha : Number (optional, default 1)
#     Smoothing parameter. E.g. if alpha equals 1, each variable in each class
#     is believed to have 1 observation by default
function MultinomialNB{C}(classes::Vector{C}, n_vars::Int64; alpha=1)
    c_counts = Dict(classes, ones(Int64, length(classes)) * alpha)
    x_counts = Dict{C, Vector{Int64}}()
    for c in classes
        x_counts[c] = ones(Int64, n_vars) * alpha
    end
    x_totals = ones(Float64, n_vars) * alpha * length(c_counts)
    MultinomialNB{C}(c_counts, x_counts, x_totals, sum(x_totals))
end


function Base.show(io::IO, m::MultinomialNB)
    print(io, "MultinomialNB($(m.c_counts))")
end


#####################################
######  Gaussian Naive Bayes  #######
#####################################

type GaussianNB{C} <: NBModel
    c_counts::Dict{C, Int64}           # count of ocurrences of each class
    c_stats::Dict{C, DataStats}        # aggregative data statistics
    gaussians::Dict{C, MvNormal}        # precomputed distribution
    # x_counts::Dict{C, Vector{Number}}  # ?? count/sum of occurrences of each var
    # x_totals::Vector{Number}           # ?? total occurrences of each var
    n_obs::Int64                       # total number of seen observations
end


function GaussianNB{C}(classes::Vector{C}, n_vars::Int64)
    c_counts = Dict(classes, zeros(Int64, length(classes)))
    c_stats = Dict(classes, [DataStats(n_vars, 2) for i=1:length(classes)])
    gaussians = Dict{C, MvNormal}()
    GaussianNB{C}(c_counts, c_stats, gaussians, 0)
end


function Base.show(io::IO, m::GaussianNB)
    print(io, "GaussianNB($(m.c_counts))")
end
