// Practical Machine learning
// Logistic Regression example
// Chapter 10

package default

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * @author Oleksiy Dyagilev
 */
object SpamClassification extends App {

  runSpark()

  def runSpark() {
    val conf = new SparkConf().setAppName("Spam classification").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val file = sc.textFile("./dataset/spambase.data")
    val examples = file.map { line =>
      val parts = line.split(",").map(_.toDouble)
      LabeledPoint(parts.last, Vectors.dense(parts.init))
    }

    val splits = examples.randomSplit(Array(0.8, 0.2))
    val training = splits(0).cache()
    val test = splits(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    examples.unpersist(blocking = false)

    val algorithm = new LogisticRegressionWithLBFGS()

    //      new SquaredL2Updater()
    val updater = new L1Updater()

    algorithm.optimizer
      .setNumIterations(1000)
      .setUpdater(updater)
//      .setRegParam(0.0)

    val model = algorithm.run(training).clearThreshold()

    val prediction = model.predict(test.map(_.features))
    val predictionAndLabel = prediction.zip(test.map(_.label))

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    println(s"Test areaUnderPR = ${metrics.areaUnderPR()}.")
    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")

    sc.stop()

  }
}
