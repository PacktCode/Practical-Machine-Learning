# Practical Machine learning
# Regression Analysis  - Logistic Regression example
# Chapter 10


## Initialization
using PyPlot
using NLopt

function costFunction(theta, X, y)

  # Initialize some useful values
  m = length(y) # number of training examples

  # You need to return the following variables correctly
  J = COSTFUNCTION(theta, X, y)
  grad = zeros(size(theta)[1])
  return (J, grad)
end

function plotData(X, y)

  # Create New Figure
  figure()
  hold(true)
  PLOTDATA(x,y)
  hold(false)
end

function plotDecisionBoundary(theta, X, y)
  # Plot Data
  plotData(X[:, 2:3], y)
  hold(true)

  if size(X, 2) <= 3
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = [minimum(X[:, 2])-2,  maximum(X[:, 2])+2]

    # Calculate the decision boundary line
    plot_y = (-1./theta[3]).*(theta[2].*plot_x + theta[1])

    # Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)

    # Legend, specific for the exercise
    # legend("Admitted", "Not admitted", "Decision Boundary")
    legend("Admitted", "Not admitted")
    axis([30, 100, 30, 100])
  else
    # Here is the grid range
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)

    z = zeros(length(u), length(v))
    # Evaluate z = theta*x over the grid
    for j in 1:length(v), i in 1:length(u)
      z[i, j] = (mapFeature([u[i]], [v[j]]) * theta)[1]
    end
    z = z' # important to transpose z before calling contour

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], linewidth=2)
  end
  hold(false)
end

function sigmoid(z)
  J = SIGMOID(z)
  return J
end

function  predict(theta, X)
  m = size(X, 1) # Number of training examples
  p = PREDICT(theta, X) 
 return p
end


## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = readcsv("dataset1.txt")
X = data[:, 1:2]
y = data[:, 3]

## ==================== Plotting ====================

@printf("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n")

plotData(X, y)

# Put some labels
hold(true)
# Labels and Legend
xlabel("Exam 1 score")
ylabel("Exam 2 score")

# Specified in plot order
legend(["Admitted", "Not admitted"])
hold(false)

@printf("Program paused. Press enter to continue.\n")
readline()


## ============ Compute Cost and Gradient ============

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = size(X)

# Add intercept term to x and X_test
X = [ones(m, 1) X]

# Initialize fitting parameters
initial_theta = zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

@printf("Cost at initial theta (zeros): %f\n", cost)
@printf("Gradient at initial theta (zeros): \n")
@printf("%s", join(map(x -> @sprintf(" %f ", x), grad), "\n"))

@printf("\nProgram paused. Press enter to continue.\n")
readline()


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
#options = optimset("GradObj", "on", "MaxIter", 400)
options = Opt(:LN_NELDERMEAD, n+1)
min_objective!(options, (theta, grad) -> costFunction(theta, X, y)[1])
maxeval!(options, 400)
#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost
(cost, theta, _) = optimize(options, initial_theta)

# Print theta to screen
@printf("Cost at theta found by optimize: %f\n", cost)
@printf("theta: \n")
@printf("%s\n", join(map(x -> @sprintf(" %f ", x), theta), "\n"))

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Put some labels
hold(true)
# Labels and Legend
xlabel("Exam 1 score")
ylabel("Exam 2 score")

# Specified in plot order
legend(["Regression", "Admitted", "Not admitted"])
hold(false)

@printf("\nProgram paused. Press enter to continue.\n")
#readline()


prob = sigmoid([1 45 85] * theta)
@printf("For a student with scores 45 and 85, we predict an admission probability of %f\n\n", prob[1])

# Compute accuracy on our training set
p = predict(theta, X)

@printf("Train Accuracy: %f\n", mean(p .== y) * 100)
