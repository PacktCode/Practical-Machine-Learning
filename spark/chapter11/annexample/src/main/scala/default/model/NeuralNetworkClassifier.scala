// Practical Machine learning
// Neural Network example
// Chapter 11

package default.model

import breeze.linalg.{argmax, DenseVector, sum}
import breeze.numerics.log



class NeuralNetworkClassifier(nGroup: Int) extends NeuralNetworkModel(){

  override def cost(activations: DenseVector[Double], y: Double): Double ={

    handelException(activations, y)

    val tmp_act: DenseVector[Double] =  ((- activations) + 1d)

    val yInt: Int = y.floor.toInt

    tmp_act(yInt) = activations(yInt)

    sum(-log(tmp_act))

  }

  override def delta(activations: DenseVector[Double], y: Double): DenseVector[Double] = {

    handelException(activations, y)

    val tmp_act: DenseVector[Double] = activations.copy

    val yInt: Int = y.floor.toInt

    tmp_act(yInt) = tmp_act(yInt) - 1d

    tmp_act
  }

  override def predict(activations: DenseVector[Double]): Double ={
    require(activations.length == nGroup, "Number of output activations is not equal to number of group.")

    val groupWithMaxActivation = argmax(activations)
    groupWithMaxActivation.toDouble
  }



  private def handelException(activations: DenseVector[Double], y: Double): Unit ={

    require(activations.length == nGroup, "Number of output activations is not equal to number of group.")

    require(0 <= y && y < nGroup, "y is out of range: " + "y=" + y + "; nGroup=" + nGroup)

  }
}
