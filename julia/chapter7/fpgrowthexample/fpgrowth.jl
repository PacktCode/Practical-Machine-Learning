# Practical Machine learning
# Association rule based learning - FPGrowth example 
# Chapter 7

abstract FPTree

type FPEmptyNode <: FPTree end

type FPTreeNode <: FPTree
    key::Int64 # I thought Generic Types should be used here, but it is much more harder to implement than I expected
    name::Int64
    cnt::Int64
    parent::Int64
    children::Array{Int64}

    function FPTreeNode(k,v)
        node = new(k,v,1,0)
        node.children = Array(Int64,0)
        node
    end

    function FPTreeNode(k,v,p)
        node = new(k,v,1,p)
        node.children = Array(Int64,0)
        node
    end

    function FPTreeNode(k,v,p,c)
        node = new(k,v,1,p)
        if typeof(c) <: Array{Int64}
            node.children = c
        else
            node.children = Array(Int64,0)
        end
        node
    end
end

function haschild(idx,childname,N)
    length(getnode(idx,N).children) < 1 && return false
    for c in getnode(idx,N).children
        getnode(c,N).name == childname && return true
    end
    return false
end

function getchildindex(idx,childname,N)
    for c in getnode(idx,N).children
        getnode(c,N).name == childname && return c
    end
end

haschild(t) = ifelse(length(t.children) < 1,false,true)

function growchild!(idx, childlist::AbstractArray, N::AbstractArray)
    nchild = length(childlist)
    if nchild < 1 return
    else
        child_idx = length(N)
        child_name = childlist[1]
        deleteat!(childlist,1)
        n = FPTreeNode(child_idx, child_name, idx)
        filter!(x -> !isempty(x), getnode(idx,N).children)
        push!(getnode(idx,N).children, child_idx)
        push!(N, n)
    end
    growchild!(child_idx, childlist,N)
end

function climb_grow(idx, childlist, N)
    isempty(childlist) && return
    childname = childlist[1]
    if haschild(idx, childname, N)
        childindex = getchildindex(idx, childname,N)
        getnode(childindex,N).cnt += 1
        deleteat!(childlist,1)
        climb_grow(childindex, childlist, N)
    else growchild!(idx, childlist,N)
    end
end

# Climb down from tree node to root, and record the path
function climb_down!(idx, N, path)
    push!(path, idx)
    idx == 0 && return sort!(path)
    climb_down!(getnode(idx,N).parent, N, path)
end

function remove_node(idx, N)
    _N = Array(FPTree,0)
    for n in N
        n.key == idx && continue
        n.parent == idx && (n.parent = 0) # if parent node is going to be removed, set root(null) node as parent
        if !isempty(n.children)
            _children = Array(Int64, 0)
            for i in 1:length(n.children)
                if n.children[i] != idx
                    push!(_children, n.children[i])
                end
            end
            n.children = _children
        end
        push!(_N, n)
    end
    _N
end

# Get a certain node by its key
function getnode(idx::Int64,N::Array{FPTree})
    for n in N
        if n.key == idx return n
        end
    end
    throw(BoundsError())
end


function gen_tree(T)
    N = Array(FPTree,0)
    # TODO: Sort T
    # Add null node
    n0 = FPTreeNode(0,0)
    push!(N,n0)
    for t in T
        climb_grow(0,t,N)
    end
#     @show N
    N
end

function test()
    T = Array(Array{Int64,1},10)
    T[1] = [1,2]
    T[2] = [2,3,4]
    T[3] = [1,3,4,5]
    T[4] = [1,4,5]
    T[5] = [1,2,3]
    T[6] = [1,2,3,4]
    T[7] = [1]
    T[8] = [1,2,3]
    T[9] = [1,2,4]
    T[10] = [2,3,5]
    N = gen_tree(T)
end

function fp_growth!(N, minsupp_cnt)
    # get itemset
    I = filter(n->n>0,unique(sort([Int64(n.name) for n in N])))
    # get the 1-consequent item. if the item is infrequent, find the preceding frequent item
    last_nodes = []
    while true
        last = pop!(I)
        last_nodes = filter(n->n.name==last && isempty(n.children), N)
        supp = 0
        for n in last_nodes
            supp += n.cnt
        end
        supp >= minsupp_cnt && break
        isempty(I) && error("No item is frequent enough in given transactions.")
    end
    # get path with the 1-consequent
    paths = []
    for l in last_nodes
        path = Array(Int64, 0)
        climb_down!(l.key,N,path)
        isempty(path) || push!(paths, path)
    end
    # prune items not in any path
    I = unique(sort(reduce(vcat,paths)))
    remove_idx = filter(i->!(i in I), map(i->i.key,N))
    for i in remove_idx
        N = remove_node(i,N)
    end
    cond_fp_tree(N, paths, minsupp_cnt)
end

function cond_fp_tree(N, paths, minsupp_cnt)
    last_item = 0
    # update support count on each path
    for n in N # set all count but last to 0
        isempty(n.children) || (n.cnt = 0)
    end
    for path in paths
        sort!(path)
        last_item == 0 && (last_item = getnode(path[end],N).name) # update last_item when not assigned
        for i in length(path)-1:-1:1
            getnode(path[i],N).cnt += getnode(path[i+1],N).cnt
        end
        N = remove_node(path[end],N)# remove the 1-consequent in paths
    end
    # prune the items that nolonger frequent
    item_cnt = Dict{Int64, Int64}()
    for n in N # collect item count
        haskey(item_cnt,n.name) || (item_cnt[n.name] = 0)
        item_cnt[n.name] += n.cnt
    end
    for (k,v) in item_cnt
        v < minsupp_cnt && delete!(item_cnt, k)
    end
    remove_idx = [n.key for n in filter(n->!(n.name in keys(item_cnt)) && n.name != 0,N)]
    for ridx in remove_idx # prune infrequent items
        N = remove_node(ridx,N)
    end
    candidates = []
    paths = []
    leaves = filter(n->isempty(n.children), N)
    isempty(leaves) && return candidates
    for l in leaves
        path = Array(Int64, 0)
        climb_down!(l.key,N,path)
        length(path) >= 2 && push!(paths, path)
    end
    paths
#     last_item
    # Extract frequent candidate
    for path in paths
        length(path) < 2 && continue
        push!(candidates, path[end-1:end])
        pop!(path)
    end


end

N = test()
#fp_growth!(N, 2)
