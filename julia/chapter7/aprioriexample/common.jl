# Common types and functions

type Rule
    P::Array{Int64} # Antecedent
    Q::Array{Int64} # Consequent
end

# Support Count: σ(x) = | {tᵢ|x ⊆ tᵢ,tᵢ∈ T}|
function σ(x, T)
    ret = 0
    for t in T
        ⊆(x,t) && (ret += 1)
    end
    ret
end

# Support of itemset x -> y, which x does not intersect y.
supp(x,y,T) = σ(∪(x,y),T)/length(T)

# Confidence of itemset x-> y, which x does not intersect y.
conf(x,y,T) = σ(∪(x,y),T)/σ(x,T)
