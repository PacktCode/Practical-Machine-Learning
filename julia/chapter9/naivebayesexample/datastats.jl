# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

using Base.BLAS

# type for collecting data statistics incrementally
type DataStats    
    x_sums::Vector{Float64}      # sum(x_i)
    cross_sums::Matrix{Float64}  # sum(x_i'*x_i) (lower-triangular matrix)
    n_obs::Uint64                # number of observations
    obs_axis::Int64              # observation axis, e.g. size(X, obs_axis)
                                 # should return number of observations
    function DataStats(n_vars, obs_axis=1)
        @assert obs_axis == 1 || obs_axis == 2
        new(zeros(Float64, n_vars), zeros(Float64, n_vars, n_vars), 0, obs_axis)
    end
end


function Base.show(io::IO, dstats::DataStats)
    print(io, "DataStats(n_vars=$(length(dstats.x_sums))," *
          "n_obs=$(dstats.n_obs),obs_axis=$(dstats.obs_axis))")
end


# Collect data statistics.
# This method may be called multiple times on different
# data samples to collect aggregative statistics.
function updatestats(dstats::DataStats, X::Matrix{Float64})
    trans = dstats.obs_axis == 1 ? 'T' : 'N'
    axpy!(1.0, sum(X, dstats.obs_axis), dstats.x_sums)
    syrk!('L', trans, 1.0, X, 1.0, dstats.cross_sums)
    dstats.n_obs += size(X, dstats.obs_axis)
    return dstats
end

function Base.mean(dstats::DataStats)
    @assert (dstats.n_obs >= 1) "At least 1 observations is requied"
    return dstats.x_sums ./ dstats.n_obs
end

function Base.cov(dstats::DataStats)
    @assert (dstats.n_obs >= 2) "At least 2 observations are requied"
    mu = mean(dstats)
    C = (dstats.cross_sums - dstats.n_obs * (mu*mu')) / (dstats.n_obs - 1)
    Base.LinAlg.copytri!(C, 'L')
    return C
end


