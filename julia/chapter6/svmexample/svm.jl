# Practical Machine learning
# Support Vector Machine example 
# Chapter 6

## Commands from https://gist.github.com/2833165
## gcc -Wall -fPIC -c *.c
## cc -shared -Wl,-soname,libtest.so -o libtest.so *.o

## Adapt to svm:
## gcc -Wall -fPIC -c *.c *.cpp
## cc -shared -Wl,-soname,libsvm.so -o libsvm.so *.o

libsvm = dlopen("reference/libsvm.so")

function svm_nodes(X::Array{Float64,2})  ## TODO:  write version for sparse X
    r = size(X)[1]
    c = size(X)[2]
    print(r)
    print(c)
    x = reshape(transpose(X), (r * c, 1))
    svn = ccall(dlsym(libsvm, :sparsify),
                     Ptr{Void},
                     (Ptr{Float64}, Int32, Int32),
                     x, r, c)
    return svn
end

function svm_problem(y::Array{Float64,1}, X::Array{Float64,2})
    svn = svm_nodes(X)
    l = int32(size(y)[1])
    svmprob = ccall(dlsym(libsvm, :_jl_svm_problem),
                    Ptr{Void},
                    (Ptr{Float64}, Int32, Ptr{Void}),
                    y, l, svn)
    return svmprob
end
function svm_problem(f::Formula, df::DataFrame)
    mm = model_matrix(f, df)
    y = mm.response[:,1]
    X = mm.model
    svmprob = svm_problem(y, X)
    return svmprob
end
svm_problem(f::Expr, df::DataFrame) = svm_problem(Formula(f), df)

const svm_type_table = {"C_SVC"=>1,
           "NU_SVC"=>2,
           "ONE_CLASS"=>3,
           "EPSILON_SVR"=>4,
           "NU_SVR"=>5}
const kern_type_table = {"LINEAR"=>1,
            "POLY"=>2,
            "RBF"=>3,
            "SIGMOID"=>4,
            "PRECOMPUTED"=>5}

function svmparameter(svm_type::String, kernel_type::String, degree::Int32,
                      gamma::Float64, coef0::Float64, cache::Float64, tolerance::Float64,
                      cost::Float64, nu::Float64, #nweights,
                      p::Float64, shrinking::Int32, probability::Int32)
    svm_type_int = get(svm_type_table, uppercase(svm_type), 0)
    kernel_type_int = get(kern_type_table, uppercase(svm_type), 0)
    svmpar = ccall(dlsym(libsvm, :_jl_svm_par),
                   Ptr{Void},
                   (Int32, Int32, Int32,
                    Float64, Float64, Float64, Float64,
                    Float64, Float64, 
                    Float64, Int32, Int32),
                   svm_type_int, kernel_type_int, degree,
                   gamma, coef0, cache, tolerance,
                   cost, nu, #nweights,
                   p, shrinking, probability)
    return svmpar
end

function svmtrain(prob, param)
    svm_model = ccall(dlsym(libsvm, :svm_train),
                      Ptr{Void},
                      (Ptr{Void}, Ptr{Void}),
                      prob, param)
end


function svmpredict(model, X)
    svn = svmnode(X)
    r = size(X)[1]
    res = Array(Float64, (r,))
    ccall(dlsym(libsvm, :_jl_svm_predict),
          Ptr{Float64},
          (Ptr{Void}, Ptr{Void}, Int32, Ptr{Float64}),
          model, svn, r, res)
    return res
end

