// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import breeze.linalg.DenseMatrix

case class CostGradient(val cost: Double, val thetaGradient: List[DenseMatrix[Double]], val n: Int)
