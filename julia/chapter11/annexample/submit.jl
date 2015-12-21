# Practical Machine learning
# Artificial Neural Network
# Chapter 11

export submit

include("../data.jl")
include("../submit.jl")

include("nnCostFunction.jl")
include("sigmoidGradient.jl")

function submit()
  parts = [
    Part(1, "Feedforward and Cost Function"),
    Part(2, "Regularized Cost Function"),
    Part(3, "Sigmoid Gradient"),
    Part(4, "Neural Network Gradient (Backpropagation)"),
    Part(5, "Regularized Gradient")
  ]
  conf = Conf("neural-network-learning",
              "Neural Networks Learning", parts, solver)

  submitWithConf(conf)
end

function solver(partId)
  # Random Test Cases
  X = reshape(3 * sin(1:1:30), (3, 10))
  Xm = reshape(sin(1:32), (16, 2)) / 5
  ym = (1 + mod(1:16, 4)')'
  t1 = sin(reshape(1:2:24, (4, 3)))
  t2 = cos(reshape(1:2:40, (4, 5)))
  t  = [t1[:] ; t2[:]]
  if partId == 1
    J, _ = nnCostFunction(t, 2, 4, 4, Xm, ym, 0)
    return @sprintf("%0.5f", J)
  elseif partId == 2
    J, _ = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5)
    return @sprintf("%0.5f", J)
  elseif partId == 3
    return join(map(x -> @sprintf("%0.5f", x), sigmoidGradient(X)), " ")
  elseif partId == 4
    J, grad = nnCostFunction(t, 2, 4, 4, Xm, ym, 0)
    return @sprintf("%0.5f ", J) * join(map(x -> @sprintf("%0.5f", x), grad), " ")
  elseif partId == 5
    J, grad = nnCostFunction(t, 2, 4, 4, Xm, ym, 1.5)
    return @sprintf("%0.5f ", J) * join(map(x -> @sprintf("%0.5f", x), grad), " ")
  end
end
