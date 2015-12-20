# Practical Machine learning
# Regression Analysis - logistic regression
# Chapter 6


install.packages("expm")
library(expm)

# Common Functions - costFunction
costFunction <- function(X, y, theta){
  
  m=length(y)
  z <- (X %*% theta)
  h <- sigmoid(z)
  
  J = -1/(m
  ) * sum(t(y) %*% log(h) + (1-t(y)) %*% log(1-h));
  grad = 1/m * (t(X) %*% (h - y))
  
  output <- list(J, grad)
  return(J)
}

#CommonFunctions - costFunctionReg
costFunctionReg <- function(X, y, theta, lambda){
  
  #% Deal with the theta(1) term, set it to '0'
  
  thetaFilt = theta
  thetaFilt[1] <- 0
  
  m=length(y)
  z <- (X %*% theta)
  h <- sigmoid(z)
  
  J = -1/m * sum(t(y) %*% log(h) + (1-t(y)) %*% log(1-h)) + (lambda/(2*m)) * (t(thetaFilt) %*% thetaFilt)
  grad = 1/m * (t(X) %*% (h - y)) + (lambda/(m) * (theta))
  
  output <- list(J, grad)
  return(J)
}

mapFeature <- function(X1, X2){
#% MAPFEATURE Feature mapping funcion to polynomial features
#%
#%   MAPFEATURE(X1, X2) maps the two input features
#%   to quadratic features used in the regularization exercise.
#%
#%   Returns a new feature array with more features, comprising of 
#%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#%
#%   Inputs X1, X2 must be the same size
#%

rcF <- as.vector(dim(X1))
nrF <- rcF[1]
ncF <- rcF[2]

Xout1 <- matrix(1, nrow = nrF, ncol = 1)
Xout2 <- matrix(1, nrow = nrF, ncol = 1)

degree = 6;
  for(i in 1:degree){
    for(j in 0:i){
      Xout1 <- ((X1 ^ (i-j)) * (X2 ^ j))
      Xout2 <- cbind(Xout2, Xout1)
   }
  }
  return(Xout2)

}

polyFeatures <- function(X, p){
  
  #%POLYFEATURES Maps X (1D vector) into the p-th power
  #%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
  #%   maps each example into its polynomial features where
  #%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
  #%
  
  
  #% You need to return the following variables correctly.
  X_poly = matrix(0, nrow(X), p);
  
  #% ====================== YOUR CODE HERE ======================
  #% Instructions: Given a vector X, return a matrix X_poly where the p-th 
  #%               column of X contains the values of X to the p-th power.
  #%
  #% 
  
  for(i in 1:p){
    
    X_poly[,i] <- (X[,2])^i
        
  }
  
  return(X_poly)
}

predict <- function(theta, X){
#%PREDICT Predict whether the label is 0 or 1 using learned logistic 
#%regression parameters theta
#%   p = PREDICT(theta, X) computes the predictions for X using a 
#%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)#

#m = size(X, 1); % Number of training examples

#% You need to return the following variables correctly
#p = zeros(m, 1);

#% ====================== YOUR CODE HERE ======================
#% Instructions: Complete the following code to make predictions using
#%               your learned logistic regression parameters. 
#%               You should set p to a vector of 0's and 1's
#%

p = sigmoid(X %*% theta) >= 0.5

}

sigmoid <- function(z){
  g <- 1/(1+exp(-z))
  return(g)
}

#Actual Logistic Regression function example 

data1 <- read.table("dataset1.txt", sep =",")

colnames(data1) <- cx("Population","Profit")

plot(data1$V1, data1$V2, pch=19, 
     col=cut(data1$V3, breaks=c(2)))

data1.m <- as.matrix(data1)
dim(data1.m)

X.ini <- data1.m[, c(1,2)]
y <- (data1.m[, 3])

plot(data1$V1, data1$V2, pch=19, 
     col=cut(data1$V3, breaks=c(2)))

rc <- as.vector(dim(X.ini))
nr <- rc[1]
nc <- rc[2]

ones <- rep(1, nr)
           
X <- cbind(ones, X.ini)
theta <- matrix(0, nrow = nc+1, ncol = 1)

costFunction(X, y, theta)

initial_theta <- rep(0,ncol(X))
wrapper <- function(theta) costFunction(theta, X=X, y=y)
optim(initial_theta, wrapper) #, gr = "L-BFGS-B", control = list(maxit = 400))

#%% =========== Part 1: Regularized Logistic Regression ============
#  %  In this part, you are given a dataset with data points that are not
#%  linearly separable. However, you would still like to use logistic 
#%  regression to classify the data points. 
#%
#%  To do so, you introduce more features to use -- in particular, you add
#%  polynomial features to our data matrix (similar to polynomial
#                                           %  regression).


data2 <- read.table("dataset2.txt", sep =",")

plot(data2$V1, data2$V2, pch=19, 
     col=cut(data2$V3, breaks=c(2)))

data2.m <- as.matrix(data2)
dim(data2.m)

X1 <- as.matrix(data2.m[, c(1)])
X2 <- as.matrix(data2.m[, c(2)])
y <- (data2.m[, 3])


X <- mapFeature(X1,X2)

rc <- as.vector(dim(X))
nr <- rc[1]
nc <- rc[2]

#% Initialize fitting parameters
initial_theta <- matrix(0, nrow = nc, ncol = 1)

#% Set regularization parameter lambda to 1
lambda = 1;

costFunctionReg(X, y, initial_theta, lambda)

#% Initialize fitting parameters
initial_theta <- matrix(0, nrow = nc, ncol = 1)

#% Optimize

initial_theta <- rep(0,ncol(X))
wrapper <- function(initial_theta) costFunctionReg(initial_theta, X=X, y=y, lambda)
op <- optim(initial_theta, wrapper, method = "L-BFGS-B")
# method = "L-BFGS-B" it did not work for default method

theta <- as.matrix(op[[1]])

#% Compute accuracy on our training set
p = predict(theta, X)

mean(p)

# PLOT
plot_x <- c(min(X[,2])-2, max(X[,2])-2)
plot_y <- (-1/theta[3,])*(theta[2,]*plot_x + theta[1,]);

plot(plot_x, plot_y)





