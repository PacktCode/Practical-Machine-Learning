// Practical Machine learning
// Neural Network example
// Chapter 11

package default

import breeze.linalg.{DenseVector=>BDV, DenseMatrix=>BDM}
import breeze.numerics.{sigmoid}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import scala.math.abs

object Util {

  def toBreeze(v: Vector): BDV[Double] = {
    new BDV(v.toArray)
  }

  def addBias(v: BDV[Double]) : BDV[Double] ={
    BDV.vertcat(BDV(1d), v)
  }

  def removeBias(v: BDV[Double]): BDV[Double] ={
    v(1 to -1)
  }

  def sigmoidGradient(v: BDV[Double]) : BDV[Double] = {
    val s = sigmoid(v)
    s :* ((-s) + 1d)
  }

  def doubleEqual(v1: Double, v2: Double, p: Double = 1e-4): Boolean = {
    abs(v1-v2) <= p
  }

}
