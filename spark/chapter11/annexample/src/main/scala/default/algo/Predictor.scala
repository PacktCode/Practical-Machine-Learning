// Practical Machine learning
// Neural Network example
// Chapter 11

package default.algo

import breeze.numerics.sigmoid
import rotationsymmetry.neuralnetwork.model.NeuralNetworkModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix=>BDM}
import rotationsymmetry.neuralnetwork.Util


object Predictor {
  def predict(features: RDD[Vector], neuralNetworkModel: NeuralNetworkModel, theta: List[BDM[Double]]): RDD[Double] ={
    features.map(x=>{
      val xVec = Util.toBreeze(x)
      val outputActivation = theta.foldLeft(xVec)((a, th)=> sigmoid( th * Util.addBias(a)))
      neuralNetworkModel.predict(outputActivation)
    })
  }

  def predict(features: Array[Vector], neuralNetworkModel: NeuralNetworkModel, theta: List[BDM[Double]]): Array[Double] ={
    features.map(x=>{
      val xVec = Util.toBreeze(x)
      val outputActivation = theta.foldLeft(xVec)((a, th)=> sigmoid( th * Util.addBias(a)))
      neuralNetworkModel.predict(outputActivation)
    })
  }

}
