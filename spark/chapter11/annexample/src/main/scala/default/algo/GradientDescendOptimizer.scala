// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import rotationsymmetry.neuralnetwork.model.{NeuralNetworkModel, Topology}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix=>BDM, DenseVector=>BDV}

import scala.util.Random


object GradientDescendOptimizer extends LoggingAbility{

  def optimize(data: RDD[LabeledPoint],
               neuralNetworkModel: NeuralNetworkModel,
               topology: Topology,
               initTheta: List[BDM[Double]],
               rate: Double,
               lambda: Double,
               normalFactor: Double,
               maxIter: Int,
               batchProp: Double = 1,
               batchSeed: Integer): GradientDescendSolution ={


    var theta = initTheta

    val costHistory: Array[Double] = new Array[Double](maxIter)

    var i: Integer = 0
    while (i < maxIter){
      val costGradient = DistCostGradientComputer.compute(data,
        theta,
        neuralNetworkModel,
        lambda,
        normalFactor,
        batchProp,
        batchSeed + i)

      costHistory(i) = costGradient.cost

      val unrolledThetaVector: BDV[Double] = new BDV(Topology.unrollTheta(theta))
      val unrolledGradientVector: BDV[Double] = new BDV(Topology.unrollTheta(costGradient.thetaGradient))

      val updatedUnrolledThetaVector: BDV[Double] = unrolledThetaVector - (unrolledGradientVector * rate)

      theta = topology.generateThetaFrom(updatedUnrolledThetaVector.toArray)
      i = i + 1
      logger.trace("Iteration: " + i + "/" + maxIter + "        Cost: " + costGradient.cost)
    }

    GradientDescendSolution(costHistory.toList, theta)

  }
}

case class GradientDescendSolution(val costHistory: List[Double], val theta: List[BDM[Double]] )
