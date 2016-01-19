// Practical Machine learning
// Neural Network example
// Chapter 11

package default.example

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.{SparkContext, SparkConf}

object MNIST {


  def processData(): Unit = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local")
    val sc = new SparkContext(conf)
    val xData=sc.textFile("x.txt")
    val xValue = xData.map(line => line.trim().split(" ").map(_.toDouble))

    val yData=sc.textFile("y.txt")
    val yValue = yData.map(line => {
      val yInt = line.trim().toInt
      yInt match {
        case 10 => 0
        case _ => yInt
      }
    })

    val data = yValue.zip(xValue).map(
      line => LabeledPoint(line._1, Vectors.dense(line._2))
    )

    saveAsLibSVMFile(data, "data.libsvm")
    sc.stop()
  }
}
