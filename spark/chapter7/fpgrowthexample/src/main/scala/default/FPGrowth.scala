// Practical Machine learning
// Association rule based learning - FPGrowth example
// Chapter 7

package default

import java.io.{PrintWriter, File}

import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.mutable.ListBuffer
import scala.compat.Platform.currentTime

/**
 * File: FPGrowth.scala
 * Description: This is a naive implementation of FPGrowth | Parallel FPGrowth for learning how to use Spark and Scala.
 * Author: Lin, Chen
 * E-mail: chlin.ecnu@gmail.com
 * Version: 2.4
 */

object FPGrowth {
  def showWarning() {
    System.err.println(
      """
        |---------------------------------------------------------------------
        |WARN:
        |Description: This is a naive implementation of FPGrowth|Parallel FPGrowth for learning how to use Spark and Scala.
        |Author: Lin, Chen
        |E-mail: chlin.ecnu@gmail.com
        |Version: 2.4
        |---------------------------------------------------------------------
      """.stripMargin)
  }

  def showError(): Unit = {
    println(
      """
        |Usage: spark-submit --class "FPGrowth" jar method supportThreshold fileName splitterPattern numSlices numGroups printResults
        |Parameters:
        |  method - Method of processing: sequential or parallel.
        |  supportThreshold - The minimum number of times a co-occurrence must be present.
        |  fileName - The name of input file. The directory of input file is "hdfs://10.1.2.71:54310/user/clin/fpgrowth/input/".
        |  splitterPattern - Regular Expression pattern used to split given string transaction in to itemsets.
        |  numSlices - Number of slices the RDD should be divided in the parallel version. Doesn't work in sequential version.
        |  numGroups - Number of groups the features should be divided in the parallel version. Doesn't work in sequential version.
        |  printResults - Whether or not print results on the terminal.
      """.stripMargin)
    System.exit(-1)
  }
  def main(args: Array[String]): Unit = {

    if(args.length != 7){
      showError()
    }

    showWarning()

    //Receive parameters from console.
    val method = args(0)
    val supportThreshold = args(1).toDouble
    val fileName = args(2)
    val splitterPattern = args(3)
    val numSlices = args(4).toInt
    val numGroups = args(5).toInt
    val printResults = args(6)

    val startTime = currentTime

    //Initialize SparkConf.
    val conf = new SparkConf()
    conf.setMaster("spark://sr471:7177").setAppName("FPGrowth").set("spark.cores.max", "128").set("spark.executor.memory", "24G")

    //Initialize SparkContext.
    val sc = new SparkContext(conf)

    //Create distributed datasets from hdfs.
    val input = sc.textFile("hdfs://sr471:54311/user/clin/fpgrowth/input/" + fileName, numSlices)
    val dataSize: Double = input.count()
    val minSupport = (dataSize * supportThreshold).toLong

    method match {
      case "sequential" => {
        //Transform RDD to Array[Array[String]].
        val transactions = input.map(line => line.split(splitterPattern)).collect()

        //Initialize FPTree and start to  mine frequent patterns from FPTree.
        val patterns = FPTree(transactions, minSupport)

        // Print results on the terminal.
        if (printResults.equals("Yes")) {
          var count: Int = 0
          for(pattern <- patterns){
            println(pattern._1 + " " + pattern._2)
            count += 1
          }
          println("---------------------------------------------------------")
          println("count = " + count)
          println("---------------------------------------------------------")
        }
        println("count = " + patterns.length)

        //Write the elements of patterns as a text file in a given directory in local filesystem.
        val path = "/home/yilan/FPGrowth/output.txt"
        val pw = new PrintWriter(new File(path))
        for(pattern <- patterns){
          pw.write(pattern._1 + " " + pattern._2 + "\r\n")
        }
        pw.close()
      }
      case "parallel" => {
        //Initialize ParallelFPGrowth and start to mine frequent patters in parallel.
        val patterns = ParallelFPGrowth(sc, input, minSupport,splitterPattern, numGroups)
        // Print results on the terminal.
        if (printResults.equals("Yes")) {
          var count:Int = 0
          for(pattern <- patterns.collect){
            println(pattern._1 + " " + pattern._2)
            count += 1
          }
          println("---------------------------------------------------------")
          println("count = " + count)
          println("---------------------------------------------------------")
        }
        //Write elements of patterns as text files in a given directory in hdfs.
        patterns.saveAsTextFile("hdfs://sr471:54311/user/clin/fpgrowth/output/")
      }
      case _ => {
        showError()
      }
    }

    val endTime = currentTime
    val totalTime: Double = endTime - startTime

    println("---------------------------------------------------------")
    println("This program totally took " + totalTime/1000 + " seconds.")
    println("---------------------------------------------------------")

    //Stop SparkContext.
    sc.stop()
  }
}

