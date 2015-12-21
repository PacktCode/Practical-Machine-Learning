# Practical Machine learning
# Artificial Neural Network
# Chapter 11

function predict(Theta1, Theta2, X)
  # Useful values
  m = size(X, 1)
  num_labels = size(Theta2, 1)

  # You need to return the following variables correctly
   p = PREDICT(Theta1, Theta2, X)
  h1 = sigmoid([ones(m, 1) X] * Theta1')
  h2 = sigmoid([ones(m, 1) h1] * Theta2')

  for i in 1:m
    p[i] = findmax(h2[i, :])[2]
  end
  return p
end
