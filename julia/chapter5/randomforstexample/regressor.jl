# Practical Machine learning
# Decision Trees based learning - Random Forests example 
# Chapter 5

type Regressor
    n_samples::Int
    n_features::Int
    n_max_features::Int
    improvements::Vector{Float64}
    oob_error::Float64
    trees::Vector{Tree}

    function Regressor(rf, x, y)
        n_samples, n_features = size(x)

        if n_samples != length(y)
            throw(DimensionMismatch(""))
        end

        n_max_features = resolve_max_features(rf.max_features, n_features)
        @assert 0 < n_max_features <= n_features

        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, improvements, NaN, trees)
    end
end

typealias RandomForestRegressor RandomForest{Regressor}

function RandomForestRegressor(;n_estimators::Int=10, max_features::Union{Integer,AbstractFloat,Symbol}=:third, max_depth=nothing, min_samples_split::Int=2)
    RandomForest{Regressor}(n_estimators, max_features, max_depth, min_samples_split, :mse)
end

function fit{T<:TabularData}(rf::RandomForestRegressor, x::T, y::AbstractVector)
    learner = Regressor(rf, x, y)
    n_samples = learner.n_samples

    # pre-allocation
    bootstrap = Array(Int, n_samples)
    sample_weight = Array(Float64, n_samples)
    oob_predict = zeros(n_samples)
    oob_count = zeros(Int, n_samples)

    for b in 1:rf.n_estimators
        rand!(bootstrap, 1:n_samples)
        set_weight!(bootstrap, sample_weight)
        example = Trees.Example{T}(x, y, sample_weight)
        tree = Trees.Tree()
        Trees.fit(tree, example, rf.criterion, learner.n_max_features, rf.max_depth, rf.min_samples_split)
        learner.trees[b] = tree

        for s in 1:n_samples
            if sample_weight[s] == 0.0
                oob_predict[s] += Trees.predict(tree, vec(x[s, :]))
                oob_count[s] += 1
            end
        end
    end

    oob_error = 0.

    for s in 1:n_samples
        if oob_count[s] == 0
            continue
        end

        avg = oob_predict[s] / oob_count[s]
        d = y[s] - avg
        oob_error += d * d
    end

    set_improvements!(learner)
    learner.oob_error = sqrt(oob_error / countnz(oob_count))
    rf.learner = learner
    return
end

function predict{T<:TabularData}(rf::RandomForestRegressor, x::T)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    n_samples = size(x, 1)
    output = Array(Float64, n_samples)
    vs = Array(Float64, rf.n_estimators)

    for i in 1:n_samples
        for b in 1:rf.n_estimators
            tree = rf.learner.trees[b]
            vs[b] = Trees.predict(tree, vec(x[i, :]))
        end
        output[i] = mean(vs)
    end

    output
end

function oob_error(rf::RandomForestRegressor)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    rf.learner.oob_error
end
