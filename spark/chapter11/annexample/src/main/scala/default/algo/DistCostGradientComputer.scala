// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{pow, sigmoid}
import rotationsymmetry.neuralnetwork.model.{Topology, NeuralNetworkModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import rotationsymmetry.neuralnetwork.Util
import scala.util.Random

object DistCostGradientComputer {


  def compute(data: RDD[LabeledPoint],
              theta: List[BDM[Double]],
              neuralNetwork: NeuralNetworkModel,
              lambda: Double,
              normalFactor: Double,
              batchProp: Double,
              batchSeed: Int): CostGradient = {

    require(lambda >= 0)
    require(normalFactor > 0)
    require(batchProp > 0 && batchProp <= 1)

    val costGradientNoPenalty =
      data.mapPartitionsWithIndex((index, d)=>
        forwardBackward(index, d, theta, neuralNetwork, normalFactor, batchProp, batchSeed))
      .reduce(costGradientReducer)

    val costPenalty = getCostPenalty(theta, lambda / (2 * costGradientNoPenalty.n))

    val gradientPenalty = getGradientPenalty(theta, lambda / (2 * costGradientNoPenalty.n))

    val newCost = costGradientNoPenalty.cost + costPenalty

    val newGradient: List[BDM[Double]] = (costGradientNoPenalty.thetaGradient zip gradientPenalty)
      .map(pair=> (pair._1 + pair._2))

    CostGradient(newCost, newGradient, costGradientNoPenalty.n)

  }

  def forwardBackward(index: Int,
                      iterator: Iterator[LabeledPoint],
                      theta: List[BDM[Double]],
                      neuralNetwork: NeuralNetworkModel,
                      normalFactor: Double,
                      batchProp: Double,
                      batchSeed: Integer): Iterator[CostGradient] ={

    /* set up VAR for output */
    var cost_sum: Double = 0
    val thetaGradient_sum : List[BDM[Double]] = theta.map(
      th => breeze.linalg.DenseMatrix.zeros[Double](th.rows, th.cols)
    )

    /* "caching" theta without bias as it is using in every iteration */
    val thetaWithoutBias: List[BDM[Double]] = theta.map(th =>{
      th(::, 1 to -1)
    })

    /* setup a random generate to Sample MiniBatch */
    val miniBatchRand = new Random(index + batchSeed)

    /* count the size of the iterator */
    var counter: Int = 0

    while (iterator.hasNext) {

      val data = iterator.next()

      /* only conduct forward/backward propagation if
       * the record is included in the mini batch. */
      if (miniBatchRand.nextFloat() <= batchProp) {


        /* Set up data for this record */
        val x = Util.toBreeze(data.features)
        val y = data.label


        /* Forward Propagation for cost
       * the z vector does not include bias
       * the activation vector include bias
       * */
        case class Stage(val z: BDV[Double], val a: BDV[Double])

        val z_a: List[Stage] = theta.scanLeft(Stage(z = null, a = Util.addBias(x)))(
          (stage, th) => {
            val new_z: BDV[Double] = th * stage.a
            val new_a: BDV[Double] = Util.addBias(sigmoid(new_z))
            Stage(new_z, new_a)
          })

        /* get the activations at the last layer. Also remove the bias
      * This activation will be used in cost and delta.
      * */
        val act: BDV[Double] = Util.removeBias(z_a.last.a)
        /* accumulate the cost. Using normalFactor to avoid overflow. */
        cost_sum += (neuralNetwork.cost(act, y) / normalFactor)


        /* Backward Propagation for theta gradient*/
        val dL = neuralNetwork.delta(act, y)

        /* Delta: derivative of cost w.r.t z (linear predictors)
      calculation is only needed when there is at least 1 hidden layer.
       */
        val delta: List[BDV[Double]] = if (theta.size > 1) {
          val z_middle: List[BDV[Double]] = z_a.drop(1).dropRight(1).map(stage => stage.z)

          (thetaWithoutBias.drop(1) zip z_middle).scanRight(dL)(
            (theta_z: (BDM[Double], BDV[Double]), d: BDV[Double]) => {
              theta_z match {
                case (thetaNoBias: BDM[Double], z: BDV[Double]) => {
                  val part: breeze.linalg.Transpose[BDV[Double]] = (d.t * thetaNoBias)
                  part.t :* Util.sigmoidGradient(z)
                }
              }
            }
          )
        }
        else {
          List[BDV[Double]](dL)
        }

        val a_front: List[BDV[Double]] = z_a.dropRight(1).map(_.a)

        val delta_a: List[(BDV[Double], BDV[Double])] = delta zip a_front

        val thetaGradient: List[BDM[Double]] = delta_a.map(da => {
          da._1 * da._2.t
        })

        /* accumulate the gradient. Using normalFactor to avoid overflow. */
        for (i <- 0 until theta.size) {
          thetaGradient_sum(i) :+= (thetaGradient(i) / normalFactor)
        }

        counter = counter + 1
      }
    }


    /* adjust the multiplicative factor of  the cost and
    gradient to be number of records, aka the size of the iterator
     */
    cost_sum = cost_sum * (normalFactor/counter)
    for (i <-0 until theta.size){
      thetaGradient_sum(i) :*= (normalFactor/counter)
    }


    Iterator(CostGradient(cost_sum, thetaGradient_sum, counter))

  }

  def costGradientReducer(cg1: CostGradient, cg2: CostGradient): CostGradient ={

    val n_total: Int = cg1.n + cg2.n
    val weight1: Double = cg1.n.toDouble / n_total
    val weight2: Double = cg2.n.toDouble / n_total
    val cost_total: Double = cg1.cost * weight1 + cg2.cost * weight2

    val zippedGradient : List[(BDM[Double], BDM[Double])]= cg1.thetaGradient zip cg2.thetaGradient
    val gradient_total : List[BDM[Double]]= zippedGradient map (_ match {
      case (cg1: BDM[Double], cg2: BDM[Double]) => {
        val new_cg1: BDM[Double] = cg1 :* weight1
        val new_cg2: BDM[Double] = cg2 :* weight2
        new_cg1 + new_cg2
      }
    })

    CostGradient(cost_total, gradient_total, n_total)

  }




  def getCostPenalty(theta: List[BDM[Double]], lambda: Double): Double = {
    val thetaWithZeroBias = getThetaWithZeroBias(theta)

    val unrolledThetaWithZeroBias = Topology.unrollTheta(thetaWithZeroBias)
    val costPenalty: Double = unrolledThetaWithZeroBias.map(d => pow(d, 2d)).sum

    costPenalty * lambda
  }

  def getGradientPenalty(theta: List[BDM[Double]], lambda: Double) : List[BDM[Double]] = {
    val thetaWithZeroBias = getThetaWithZeroBias(theta)

    thetaWithZeroBias.map( th => {
      th :* (2 * lambda)
    })

  }

  def getThetaWithZeroBias(theta: List[BDM[Double]]): List[BDM[Double]] ={
    theta.map(th => {
      val new_th = th.copy
      new_th(::, 0) := 0d
      new_th
    })
  }



}
