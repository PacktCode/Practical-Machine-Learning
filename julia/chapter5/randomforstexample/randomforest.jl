# Practical Machine learning
# Decision Trees based learning - Random Forests example 
# Chapter 5

import .Trees: TabularData, Tree, Node, Leaf, Criterion, Gini, CrossEntropy, MSE, getroot, getleft, getright, isnode, n_samples, impurity

type RandomForest{T}
    # parameters
    n_estimators::Int
    max_features::Any
    max_depth::Int
    min_samples_split::Int
    criterion::Criterion

    # learner
    learner::Union{T,Void}

    function RandomForest(n_estimators, max_features, max_depth, min_samples_split, criterion)
        if n_estimators < 1
            error("n_estimators is too small (got: $n_estimators)")
        end

        if isa(max_features, Integer) && max_features < 1
            error("max_features is too small (got: $max_features)")
        elseif isa(max_features, AbstractFloat) && !(0. < max_features <= 1.)
            error("max_features should be in (0, 1] (got: $max_features)")
        elseif isa(max_features, Symbol) && !(is(max_features, :sqrt) || is(max_features, :third))
            error("max_features should be :sqrt or :third (got: $max_features)")
        end

        if is(max_depth, nothing)
            max_depth = typemax(Int)
        elseif isa(max_depth, Integer) && max_depth <= 1
            error("max_depth is too small (got: $max_depth)")
        end

        if min_samples_split <= 1
            error("min_sample_split is too small (got: $min_samples_split)")
        end

        if is(criterion, :gini)
            criterion = Gini
        elseif is(criterion, :entropy)
            criterion = CrossEntropy
        elseif is(criterion, :mse)
            criterion = MSE
        else
            error("criterion is invalid (got: $criterion)")
        end

        new(n_estimators, max_features, max_depth, min_samples_split, criterion, nothing)
    end
end

function resolve_max_features(max_features::Any, n_features::Int)
    if is(max_features, :sqrt)
        floor(Int, sqrt(n_features))
    elseif is(max_features, :third)
        div(n_features, 3)
    elseif isa(max_features, Integer)
        max(max_features, n_features)
    elseif isa(max_features, AbstractFloat)
        floor(Int, n_features * max_features)
    elseif is(max_features, nothing)
        n_features
    else
        error("max_features is invalid: $max_features")
    end
end

function feature_importances(rf::RandomForest)
    if is(rf.learner, nothing)
        error("not yet trained")
    end
    rf.learner.improvements
end

function set_improvements!(learner)
    improvements = learner.improvements

    for tree in learner.trees
        root = getroot(tree)
        add_improvements!(tree, root, improvements)
    end
    normalize!(improvements)
end

function add_improvements!(tree::Tree, node::Node, improvements::Vector{Float64})
    left = getleft(tree, node)
    right = getright(tree, node)
    n_left_samples = n_samples(left)
    n_right_samples = n_samples(right)
    averaged_impurity = (impurity(left) * n_left_samples + impurity(right) * n_right_samples) / (n_left_samples + n_right_samples)
    improvement = impurity(node) - averaged_impurity
    improvements[node.feature] += improvement

    add_improvements!(tree, left, improvements)
    add_improvements!(tree, right, improvements)
    return
end

function add_improvements!(::Tree, ::Leaf, ::Vector{Float64})
    # do nothing!
    return
end
