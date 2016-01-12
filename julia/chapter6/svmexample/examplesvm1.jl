# Practical Machine learning
# Support Vector Machine example 
# Chapter 6

reposDir = EnvHash()["JuliaRepos"]

## Load julia-svm
load("svm.jl")

n = int(1e3)
p = 20
X = rand(n, p)
y = float(randi((0, 1), n))

svp = svmproblem(y, X)
svparam = svmparameter("epsilon_svr", "rbf", int32(3),
                       1., 0., 40., 0.001,
                       1., 0.5, 
                       1., int32(1), int32(0))
model = svmtrain(svp, svparam)

X2 = rand(10, p)

pred = svmpredict(model, X2)
