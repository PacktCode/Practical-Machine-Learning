# Practical Machine learning
# Association rule based learning - Apriori example 
# Chapter 7

# Find k-freq-itemset in given transactions of items queried together
using StatsBase

# Find frequent itemsets from transactions
# @T: transaction list
# @minsupp: minimum support
function find_freq_itemset(T, minsupp)
    N = length(T)
    # Find itemset I from transaction list T
    I = Array(Int64,0)
    for t in T
        for i in t
            push!(I,i)
        end
    end
    I = Set(I)

    # Find freq-itemset when k = 1: Fₖ = {i | i ∈ I^σ({i}) ≥ N × minsupp}
    k = 1
    F = []
    push!(F,map(x->[x],filter(i->σ(i,T) >= N * minsupp, I))) # F₁
    while true
        Cₖ = gen_candidate(F[end]) # Generate candidate set Cₖ from Fₖ₋₁
        Fₖ = filter(c->σ(c,T) >= Nbumanzu * minsupp, Cₖ)
        if !isempty(Fₖ)
            push!(F,Fₖ) # Eliminate infrequent candidates, then set to Fₖ
        else break
        end
    end
    F
end

# Generate freq-itemset from a list of itemsets
# @x: list of itemsets
function gen_candidate(x)
    n = length(x)
    Cₖ = Array(Array{Int64,1},0)
    for a = 1:n, b = 1:n
        if a >= b;continue
        end
        is_candidate = true
        sort!(x[a]); sort!(x[b])
        for i in 1:length(x[1])-1
            if x[a][i] == x[b][i]; continue
            else is_candidate = false; break
            end
        end
        if is_candidate
            push!(Cₖ, sort!([ x[a][1:end-1], x[a][end], x[b][end] ]))
        end
    end
    Cₖ
end

# Generate rules from frequent itemsets
# @x: list of frequent itemsets
# @T: Transaction list
function gen_rules(x, T)
    if length(x) <= 1; return [] # F contains 1-itemsets only, hence no rules generated.
    end
    x = reduce(append!,x[2:end])
    R = Array(Rule,0)
    for f in x # f as each freq-f-itemset fₖ
        ap_genrules!(R,f,map(i->Array([i]),f),T) # H₁ itemset is same as f
    end
    R
end

function ap_genrules!(R, f, H, T)
    k, m = length(f), length(H[1])
    if k > m + 1
        H = gen_candidate(H)
        H_plus_1 = []
        for h in H
            p = setdiff(f,h)
            if conf(p, h, T) >= minconf
                push!(R, Rule(p,h))
                push!(H_plus_1, h)
            end
        end
        ap_genrules(R, f, H_plus_1, T)
    end
end
