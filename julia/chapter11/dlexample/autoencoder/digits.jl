# Practical Machine learning
# Deep learning - Sparse autoencoder example 
# Chapter 11

using SparseAutoencoder
using MAT

vars = matread("./data/mnist-images.mat")
data = vars["images"]

visiblesize = 28*28
hiddensize = 196
sparsityparameter = 0.1
lambda = 3e-3
beta = 3.0
patches = data[:,1:10000]

minf,W1,W2,b1,b2 = autoencode(patches,hiddensize,visiblesize,lambda=lambda,beta=beta,rho=sparsityparameter)

using HDF5, JLD
@save "./digits-results.jld"
