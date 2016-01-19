// Practical Machine learning
// Neural Network example
// Chapter 11

package default.model

import breeze.linalg.DenseMatrix
import breeze.numerics._

import scala.util.Random


class Topology(val self: List[Int]) {
  require(self.size >= 2, "Neural Network is less than 2 layers.")

  require(self.forall(_ >0), "The number of activations in a layout should be positive.")


  private[this] val rowDim = self.tail

  private[this] val colDim = self.dropRight(1) map (_ + 1)

  private[this]val dimPair =  (rowDim zip colDim) map (rc => RowColPair(rc._1, rc._2))

  private[this] val start = dimPair.scanLeft(0) (
    (s: Int, pair: RowColPair) => s + pair.row * pair.col
  )

  def generateThetaFrom(values: Array[Double]): List[DenseMatrix[Double]] ={
    require(values.length == start.last, "input is of incorrect length.")

    val dimPair_start = dimPair zip start.dropRight(1)

    dimPair_start.map(
      _ match {
        case (pair: RowColPair, s: Int) =>{
          new DenseMatrix(pair.row, pair.col, values.slice(s, s + pair.row * pair.col))
        }
      }
    )
  }

  def generateThetaFrom(rand: Random): List[DenseMatrix[Double]] ={

    val dimPair_start = dimPair zip start.dropRight(1)

    dimPair_start.map(
      _ match {
        case (pair: RowColPair, s: Int) =>{
          val eps = sqrt(6d / (pair.row + pair.col - 1))

          val value: Array[Double] = (for (i <- 0 until pair.row * pair.col) yield (rand.nextDouble() - 0.5) * 2 * eps).toArray
          new DenseMatrix(pair.row, pair.col, value)
        }
      }
    )
  }

  private[this] case class RowColPair(val row: Int, val col: Int)

}

object Topology {
  def apply(theta: List[DenseMatrix[Double]]): Topology ={
    new Topology((theta.head.cols - 1) +: theta.map(_.rows))
  }

  def unrollTheta(theta: List[DenseMatrix[Double]]): Array[Double] = {
    val unrolledMatrixList: List[List[Double]] = theta map (_.toArray.toList)
    unrolledMatrixList reduce(_ ::: _) toArray
  }
}
