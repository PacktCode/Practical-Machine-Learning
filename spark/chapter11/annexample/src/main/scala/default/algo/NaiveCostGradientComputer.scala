// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.sigmoid
import rotationsymmetry.neuralnetwork.model.{NeuralNetworkModel, Topology}
import org.apache.spark.mllib.regression.LabeledPoint
import rotationsymmetry.neuralnetwork.Util


object NaiveCostGradientComputer {

  def compute_cost(data: List[LabeledPoint], theta: List[BDM[Double]], neuralNetworkModel: NeuralNetworkModel): Double = {
    val costList = data map {d =>
      val acc = theta.foldLeft(Util.toBreeze(d.features))(
        (a, th)=>  sigmoid (th * Util.addBias(a))
      )
      neuralNetworkModel.cost(acc, d.label)
    }
    costList.sum / data.size.toDouble
  }

  def compute_gradient(data: List[LabeledPoint],
                   theta: List[BDM[Double]],
                   neuralNetworkModel: NeuralNetworkModel,
                   eps: Double): BDV[Double] ={

    val topology = Topology(theta)

    val thetaUnrolled: Array[Double] = Topology.unrollTheta(theta)

    val thetaUnrolledWithEps: List[Array[Double]] = addEps(thetaUnrolled, eps)

    val thetaWithEps = thetaUnrolledWithEps map (topology.generateThetaFrom(_))

    val costWithEps: List[Double] = thetaWithEps map (th =>
      compute_cost(data, th, neuralNetworkModel)
      )

    val costAtOrigin = compute_cost(data, theta, neuralNetworkModel)

    val diff = (BDV(costWithEps.toArray) - costAtOrigin)

    diff / eps

  }

  def addEps(thetaUnrolled: Array[Double], eps: Double): List[Array[Double]] ={
    val out = for (i <- 0 until thetaUnrolled.length) yield {
      val tmp = thetaUnrolled.clone()
      tmp(i) = tmp(i) + eps
      tmp
    }
    out.toList
  }

}
