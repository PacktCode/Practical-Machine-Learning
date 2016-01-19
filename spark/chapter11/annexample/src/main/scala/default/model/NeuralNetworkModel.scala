// Practical Machine learning
// Neural Network example
// Chapter 11

package default.model

import breeze.linalg.{DenseVector => BDV}


abstract class NeuralNetworkModel extends  Serializable {
  def cost(activations: BDV[Double], y: Double): Double
  def delta(activations: BDV[Double], y: Double): BDV[Double]
  def predict(activations: BDV[Double]): Double

}
