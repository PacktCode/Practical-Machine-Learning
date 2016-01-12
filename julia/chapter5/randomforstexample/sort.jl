# Practical Machine learning
# Decision Trees based learning - Random Forests example 
# Chapter 5

import Base: sort!

const SMALL_THRESHOLD = 40

function sort!(samples::AbstractVector, feature::AbstractVector, range::UnitRange{Int})
    # inplace sort `samples` and `feature` vector in one shot, along with `feature`, between `range`
    len = length(range)
    @assert len == length(feature) <= length(samples)
    sort!(sub(samples, range), feature, 1, endof(feature))
end

function sort!(x::AbstractVector, y::AbstractVector, lo::Int, hi::Int)
    # if the range subject to sorting is small, the insertion sort would be faster
    if hi - lo <= SMALL_THRESHOLD
        isort!(x, y, lo, hi)
    else
        # quick sort
        p = partition(x, y, lo, hi)
        sort!(x, y, lo, p - 1)
        sort!(x, y, p + 1, hi)
    end
    return
end

# insertion sort
function isort!(x::AbstractVector, y::AbstractVector, lo::Int, hi::Int)
    @inbounds for i in lo+1:hi
        elm = y[i]
        tmp = x[i]
        j = i
        while j > lo && y[j-1] > elm
            # shift elements
            y[j] = y[j-1]
            x[j] = x[j-1]
            j -= 1
        end
        y[j] = elm
        x[j] = tmp
    end
end

function median(x::AbstractVector, i::Int, j::Int, k::Int)
    if x[i] < x[j]
        if x[j] < x[k]
            return j
        elseif x[k] < x[i]
            return i
        else
            return k
        end
    else
        # implies x[j] <= x[i]
        if x[k] <= x[j]
            return j
        elseif x[i] <= x[k]
            return i
        else
            return k
        end
    end
end

function partition(x::AbstractVector, y::AbstractVector, lo::Int, hi::Int)
    # choose pivot
    pivot_index = median(y, lo, hi, div(lo + hi, 2))
    pivot_value = y[pivot_index]

    # swap elements at pivot_index and hi
    y[pivot_index], y[hi] = y[hi], y[pivot_index]
    x[pivot_index], x[hi] = x[hi], x[pivot_index]

    p = lo
    @inbounds for i in lo:hi-1
        if y[i] <= pivot_value
            # swap elements at i and p
            y[i], y[p] = y[p], y[i]
            x[i], x[p] = x[p], x[i]
            p += 1
        end
    end

    # swap elements at p and hi
    y[p], y[hi] = y[hi], y[p]
    x[p], x[hi] = x[hi], x[p]

    p
end
