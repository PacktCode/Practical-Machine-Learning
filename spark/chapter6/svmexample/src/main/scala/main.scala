// Practical Machine learning
// Support Vector machine example 
// Chapter 6

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._

import org.apache.spark.mllib.util.MLUtils

import java.io._
import java.lang.System

object TestKernelSVM {
    def main(args: Array[String]) {
      
      if (args.length != 1 ) {
      println("Usage: /path/to/spark/bin/spark-submit --packages amplab:spark-indexedrdd:0.1" +
        "target/scala-2.10/spark-kernel-svm_2.10-1.0.jar <data file>")
      sys.exit(1)
      }
      
      val logFile = "README.md" // Should be some file on your system
      val conf = new SparkConf().setAppName("KernelSVM Test")
      val sc = new SparkContext(conf)
       
      val data =  MLUtils.loadLibSVMFile(sc, args(0))

      val splits = data.randomSplit(Array(0.8,0.2))
      val training = splits(0)
      val test = splits(1).collect()
      
      val m = training.count()
      
      var pack_size = 100
      
      val iterations = List((0.5*m).toLong,m.toLong,(1.5*m).toLong,(2*m).toLong)
      var num_iter = 0
      
      val pw = new PrintWriter(new File("result.txt" ))
      
      for (num_iter <- iterations) {
        val t1 = System.currentTimeMillis
        val svm = new KernelSVM(training, 1.0/m, "rbf", 1.0)
        svm.train(num_iter,pack_size)
        val t2 = System.currentTimeMillis
        val runtime = (t2 - t1)/1000
        
        var ss = m.toString + " " + num_iter.toString + " " + pack_size.toString + " " + svm.getAccuracy(test).toString + " " + runtime.toString + "\n"
        pw.write(ss)
      }
      
      pw.close
    }
}
