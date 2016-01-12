# Practical Machine learning
# Decision Trees based learning - Random Forests example 
# Chapter 5

typealias TabularData Union{AbstractMatrix,DataFrame}

type Example{T<:TabularData}
    x::T  # tabular data
    y::AbstractVector
    n_labels::Int
    n_features::Int
    sample_weight::Vector{Float64}

    function Example(x::T, y::AbstractVector, sample_weight::Vector{Float64})
        n_labels = length(unique(y))
        n_features = size(x, 2)
        new(x, y, n_labels, n_features, sample_weight)
    end

    Example(x::T, y::AbstractVector) = Example{T}(x, y, ones(Float64, size(x, 1)))
end
