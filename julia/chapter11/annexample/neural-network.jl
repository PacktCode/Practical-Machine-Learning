# Practical Machine learning
# Artificial Neural Network
# Chapter 11

## Initialization

using PyCall, NLopt

@pyimport scipy.io as si

include("displayData.jl")
include("nnCostFunction.jl")
include("randInitializeWeights.jl")
include("checkNNGradients.jl")
include("predict.jl")
# include("fmincg.jl")

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

## =========== Loading and Visualizing Data =============

# Load Training Data
@printf("Loading and Visualizing Data ...\n")

data = si.loadmat("ex4data1.mat")
X = data["X"]
y = data["y"]
m = size(X, 1)

# Randomly select 100 data points to display
sel = randperm(size(X, 1))
sel = sel[1:100]

displayData(X[sel, :])

@printf("Program paused. Press enter to continue.\n")
readline()


## ================  Loading Parameters ================

@printf("\nLoading Saved Neural Network Parameters ...\n")

# Load the weights into variables Theta1 and Theta2
nn = si.loadmat("ex4weights.mat")
Theta1 = nn["Theta1"]
Theta2 = nn["Theta2"]

# Unroll parameters
nn_params = [Theta1[:] ; Theta2[:]]

## ================ Compute Cost (Feedforward) ================
\@printf("\nFeedforward Using Neural Network ...\n")

# Weight regularization parameter (we set this to 0 here).
lambda = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda)

println(J)
@printf("Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.287629)\n", J)

@printf("\nProgram paused. Press enter to continue.\n")
readline()

## =============== Implement Regularization ===============

@printf("\nChecking Cost Function (w/ Regularization) ... \n")

# Weight regularization parameter (we set this to 1 here).
lambda = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda)

@printf("Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.383770)\n", J)

@printf("Program paused. Press enter to continue.\n")
readline()


## ================ Sigmoid Gradient  ================

@printf("\nEvaluating sigmoid gradient...\n")

g = sigmoidGradient([1 -0.5 0 0.5 1])
@printf("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ")
@printf("%s", join(map(x -> @sprintf("%5f", x), g), " "))
@printf("\n\n")

@printf("Program paused. Press enter to continue.\n")
readline()

## ================ Initializing Parameters ================

@printf("\nInitializing Neural Network Parameters ...\n")

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = [initial_Theta1[:] ; initial_Theta2[:]]


## =============== Implement Backpropagation ===============

@printf("\nChecking Backpropagation... \n")

#  Check gradients by running checkNNGradients
checkNNGradients()

@printf("\nProgram paused. Press enter to continue.\n")
readline()


## =============== Implement Regularization ===============

@printf("\nChecking Backpropagation (w/ Regularization) ... \n")

#  Check gradients by running checkNNGradients
lambda = 3
checkNNGradients(lambda)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size,
                             hidden_layer_size, num_labels, X, y, lambda)

@printf("""

  Cost at (fixed) debugging parameters (w/ lambda = 10): %f
  (this value should be about 0.576051)

""", debug_J)

@printf("Program paused. Press enter to continue.\n")

## =================== Training NN ===================
@printf("\nTraining Neural Network... \n")


# options = optimset('MaxIter', 50)
options = Opt(:LD_TNEWTON, length(initial_nn_params))
maxeval!(options, 50)

#  You should also try different values of lambda
lambda = 1

# Create "short hand" for the cost function to be minimized
function costFunction(theta, grad)
  cost, gs = nnCostFunction(theta, input_layer_size, hidden_layer_size,
                            num_labels, X, y, lambda)
  if length(grad) > 0
    grad[:] = gs
  end
  return cost
end

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
# [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

min_objective!(options, costFunction)
(cost, theta, _) = optimize(options, initial_nn_params)

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)],
                 hidden_layer_size, (input_layer_size + 1))

Theta2 = reshape(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end],
                 num_labels, (hidden_layer_size + 1))

@printf("Program paused. Press enter to continue.\n")
readline()

## =================  Visualize Weights =================

@printf("\nVisualizing Neural Network... \n")

displayData(Theta1[:, 2:end])

@printf("\nProgram paused. Press enter to continue.\n")
readline()

## ================= Implement Predict =================

pred = predict(Theta1, Theta2, X)

@printf("\nTraining Set Accuracy: %f\n", mean(pred .== y) * 100)
