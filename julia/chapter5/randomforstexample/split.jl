# Practical Machine learning
# Decision Trees based learning - Random Forests example 
# Chapter 5

# Criteria
abstract Criterion
abstract ClassificationCriterion <: Criterion
abstract RegressionCriterion <: Criterion

immutable GiniCriterion <: ClassificationCriterion; end  # Gini index
immutable CrossEntropyCriterion <: ClassificationCriterion; end  # cross entropy
immutable MSECriterion <: RegressionCriterion; end  # Mean Sequared Error
const Gini = GiniCriterion()
const CrossEntropy = CrossEntropyCriterion()
const MSE = MSECriterion()

# Splitters
abstract Splitter

type ClassificationSplitter{T<:ClassificationCriterion} <: Splitter
    samples::Vector{Int}
    feature::AbstractVector
    range::UnitRange{Int}
    example::Example

    n_left_samples::Float64
    n_right_samples::Float64
    left_label_counts::Vector{Float64}
    right_label_counts::Vector{Float64}

    function ClassificationSplitter(samples, feature, range, example)
        left_label_counts = zeros(example.n_labels)
        right_label_counts = zeros(example.n_labels)
        n_right_samples = 0.
        for i in 1:endof(range)
            s = samples[range[i]]
            w = example.sample_weight[s]
            l = example.y[s]
            n_right_samples += w
            right_label_counts[l] += w
        end
        new(samples, feature, range, example, 0., n_right_samples, left_label_counts, right_label_counts)
    end
end

type RegressionSplitter{T<:RegressionCriterion} <: Splitter
    samples::Vector{Int}
    feature::AbstractVector
    range::UnitRange{Int}
    example::Example

    n_left_samples::Float64
    n_right_samples::Float64
    sum_left::Float64
    sum_right::Float64
    sqsum_left::Float64
    sqsum_right::Float64

    function RegressionSplitter(samples, feature, range, example)
        n_right_samples = 0.
        sum_right = 0.
        sqsum_right = 0.
        for i in 1:endof(range)
            s = samples[range[i]]
            w = example.sample_weight[s]
            v = example.y[s]
            n_right_samples += w
            mass = v * w
            sum_right += mass
            sqsum_right += v * mass
        end
        new(samples, feature, range, example, 0., n_right_samples, 0., sum_right, 0., sqsum_right)
    end
end

immutable Split{T}
    threshold::T
    boundary::Int
    left_range::Range{Int}
    right_range::Range{Int}
    n_left_samples::Float64
    n_right_samples::Float64
    left_impurity::Float64
    right_impurity::Float64
end

Base.start(sp::Splitter) = 1

function Base.done(sp::Splitter, state::Int)
    # constant feature
    sp.feature[state] == sp.feature[end]
end

function Base.next(sp::ClassificationSplitter, state::Int)
    # seek for the next boundary
    local threshold, boundary = 0
    @inbounds for i in state:endof(sp.range)-1
        # transfer sample `s` from right to left
        s = sp.samples[sp.range[i]]
        w = sp.example.sample_weight[s]
        # as for classification the type of a label must be Int
        # this type annotation is useful for improving performance
        l = sp.example.y[s]::Int
        sp.n_left_samples += w
        sp.n_right_samples -= w
        sp.left_label_counts[l] += w
        sp.right_label_counts[l] -= w

        if sp.feature[i] != sp.feature[i+1]
            boundary = i
            threshold = sp.feature[i]
            break
        end
    end

    @assert boundary > 0

    left_range = sp.range[1:boundary]
    right_range = sp.range[boundary+1:end]
    left_impurity, right_impurity = impurity(sp)
    Split(threshold, boundary, left_range, right_range, sp.n_left_samples, sp.n_right_samples, left_impurity, right_impurity), boundary + 1
end

function Base.next(sp::RegressionSplitter, state::Int)
    # seek for the next boundary
    local threshold, boundary = 0
    @inbounds for i in state:endof(sp.range)-1
        # transfer sample `s` from right to left
        s = sp.samples[sp.range[i]]
        w = sp.example.sample_weight[s]
        v = sp.example.y[s]
        sp.n_left_samples += w
        sp.n_right_samples -= w
        mass = v * w
        sp.sum_left += mass
        sp.sum_right -= mass
        mass *= v  # mass = (v * v) * w
        sp.sqsum_left += mass
        sp.sqsum_right -= mass

        if sp.feature[i] != sp.feature[i+1]
            boundary = i
            threshold = sp.feature[i]
            break
        end
    end

    @assert boundary > 0

    left_range = sp.range[1:boundary]
    right_range = sp.range[boundary+1:end]
    left_impurity, right_impurity = impurity(sp)
    Split(threshold, boundary, left_range, right_range, sp.n_left_samples, sp.n_right_samples, left_impurity, right_impurity), boundary + 1
end

function count_labels(samples::Vector{Int}, example::Example)
    counts = zeros(Float64, example.n_labels)
    n_samples = 0.
    for s in samples
        n_samples += example.sample_weight[s]
        label = example.y[s]
        counts[label] += example.sample_weight[s]
    end

    counts, n_samples
end

function impurity(sp::ClassificationSplitter{GiniCriterion})
    left_impurity = 0.
    right_impurity = 0.

    for i in 1:endof(sp.left_label_counts)
        r = sp.left_label_counts[i] / sp.n_left_samples
        left_impurity += r * r

        r = sp.right_label_counts[i] / sp.n_right_samples
        right_impurity += r * r
    end

    1. - left_impurity, 1. - right_impurity
end

function impurity(samples::Vector{Int}, example::Example, ::GiniCriterion)
    counts, n_samples = count_labels(samples, example)
    gini_index = 0.
    for c in counts
        r = c / n_samples
        gini_index += r * r
    end
    1. - gini_index
end

function impurity(sp::ClassificationSplitter{CrossEntropyCriterion})
    left_impurity = 0.
    right_impurity = 0.

    for i in 1:endof(sp.left_label_counts)
        p = sp.left_label_counts[i] / sp.n_left_samples
        left_impurity -= p == 0. ? 0. : p * log(p)

        p = sp.right_label_counts[i] / sp.n_right_samples
        right_impurity -= p == 0. ? 0. : p * log(p)
    end

    left_impurity, right_impurity
end

function impurity(samples::Vector{Int}, example::Example, ::CrossEntropyCriterion)
    counts, n_samples = count_labels(samples, example)
    cross_entropy = 0.
    for c in counts
        p = c / n_samples
        if p != 0.
            cross_entropy -= p * log(p)
        end
    end
    cross_entropy
end

function impurity(sp::RegressionSplitter{MSECriterion})
    left_mean = sp.sum_left / sp.n_left_samples
    right_mean = sp.sum_right / sp.n_right_samples

    # sample variance
    (sp.sqsum_left / sp.n_left_samples - left_mean * left_mean,
     sp.sqsum_right / sp.n_right_samples - right_mean * right_mean)
end

function impurity(samples::Vector{Int}, example::Example, ::MSECriterion)
    n_samples = 0.
    sum = 0.
    sqsum = 0.

    for s in samples
        n_samples += example.sample_weight[s]
        v = example.y[s]
        sum += v
        sqsum += v * v
    end

    mean = sum / n_samples
    sqsum / n_samples - mean * mean
end
