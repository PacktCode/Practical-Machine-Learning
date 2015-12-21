# Practical Machine learning
# Artificial Neural Network
# Chapter 11

function debugInitializeWeights(fan_out, fan_in)
  #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
  #   the first row of W handles the "bias" terms

  # Set W to zeros
  W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out)

  # Initialize W using "sin", this ensures that W is always of the same
  # values and will be useful for debugging
  W = reshape(sin(1:length(W)), size(W)) / 10

  return W
end
