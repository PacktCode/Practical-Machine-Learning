// Practical Machine learning
// Association rule based learning - Apriori example
// Chapter 7

package default

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.SortedSet

/**
 * Delete me and write real tests.
 */
object TestMain extends App with FrequentItemSets {

  val conf: SparkConf = new SparkConf().setMaster("local").setAppName("Simple Application")
  val sparkContext: SparkContext = new SparkContext(conf)

  val filePath: String = "/home/shashir/data.txt"

  val data: RDD[ItemSet[String]] = sparkContext.textFile(filePath, 2).map { line: String =>
    SortedSet(line.split(" "): _*)
  }.cache()

  APriori(sparkContext)(data, 400, 3).foreach(println)
}
