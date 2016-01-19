// Practical Machine learning
// Clustering based learning - K-Means clustering example
// Chapter 8


package default

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
import org.apache.spark.Accumulator
import org.apache.spark.storage._
import org.apache.spark.Partitioner

import scala.collection.mutable
import scala.collection.Map

object JobDriver {

	val pointDelim = " ";
	val centroidDelim = "\t";
	var sc:SparkContext = null;

	/**
	 * Calculate the euclidean distances between two points represented as arrays.
	 */
	def distance(p1:Array[Double], p2:Array[Double]):Double = {
		Math.sqrt((p1, p2).zipped.map { (d1,d2) => Math.pow(d1-d2, 2) }.sum)
	}

	/**
	 * Calculate the partition of the assigned points based on their assigned
	 * centroid Id.
	 */
	class CentroidPartitioner[V](partitions: Int) extends Partitioner {
		def numPartitions: Int = {
			partitions
		}

		def getPartition(key: Any): Int = {
			val k = key.asInstanceOf[Int]
			return k % partitions
		}
	}

	/**
	 * Step 1 - Load the points data in a cached RDD.
	 */
	def loadPoints(pIn:String, numPartitions:Int) = {
		// Done once at the start of the algorithm. Ensure that the deserialized
		// object representations are used.
		sc.textFile(pIn, numPartitions).map { _.split(pointDelim).map(_.toDouble) }
	}

	/**
	 * Step 2 - Load the centroids data in a cached RDD.
	 */
	def loadCentroids(cIn:String) = {
		// This is only done once at the start of the algorithm. Ensures that the
		// desearialized object representations are used.
		sc.textFile(cIn).map { l =>
			val lidx = l.indexOf(centroidDelim)
			val k = l.substring(0, lidx).toInt
			val v = l.substring(lidx + 1, l.length).split(pointDelim).map(_.toDouble)
			k -> v
		}
	}

	/**
	 * Step 3 - Assign each input data point to its closest centroid label.
	 */
	def assign(centroids:Broadcast[Map[Int, Array[Double]]], points:RDD[Array[Double]]) = {
		points.map { point =>
			var closestDistance: Double = 0.0
			var closestCentroid: Int = -1
			// Find the closest centroid.
			for ((label, centroid) <- centroids.value) {
				var dist = distance(centroid, point)
				if (-1 == closestCentroid || closestDistance > dist) {
					closestDistance = dist
					closestCentroid = label
				}
			}
			// Emit the closest centroid id, with the point and the inital number of
			// points the array represents - i.e. 1.
			closestCentroid -> (point, 1)
		}
	}

	/**
	* Step 4 - Calculate the new candidate centroids.
	*/
	def calculate(assigned:RDD[(Int, (Array[Double], Int))], centroids:Broadcast[Map[Int, Array[Double]]], converged:Accumulator[Int], delta:Double) = {
		var numCentroids = centroids.value.size
		var partition = new CentroidPartitioner(numCentroids)
		// Build a custom reduce function so we can use the partitioner.
		def reducer(x:Tuple2[Array[Double],Int],
								y:Tuple2[Array[Double],Int]):Tuple2[Array[Double],Int] = (x,y) match {
			case ((pnt1,sum1),(pnt2,sum2)) =>
				var newPoint = (pnt1, pnt2).zipped.map(_+_)
				var newSum = sum1 + sum2
				(newPoint, newSum)
		}
		// Use the custom partitioner and the reduce function for the calculation of
		// the new centroids.
		assigned.reduceByKey(partition, reducer _).map { case (label, (partial, n)) =>
			var newCentroid = partial.map(_ / n)
			var oldCentroid = centroids.value.get(label).get
			// Count any centroids with the same Id that are too close.
			if (distance(newCentroid, oldCentroid) > delta) {
				converged += 1
			}
			label -> newCentroid
		}
	}

	/**
	 * Run the algorithm until the centroids have converged.
	 */
	def run(pIn:String, cIn:String, cOut:String, max:Int, delta:Double, partitions:Int) = {
		var iteration = 1
		var hasConverged = false
		// Step 1
		val points = loadPoints(pIn, partitions).cache()
		// Step 2
		var centroids = loadCentroids(cIn).collectAsMap()
		while (iteration <= max && !hasConverged) {
			printf("Starting Iteration %s\n", iteration)
			// Step 3
			var broadcast = sc.broadcast(centroids)
			var accumulator = sc.accumulator(0)
			var assigned = assign(broadcast, points)
			// Step 4
			var candidates = calculate(assigned, broadcast, accumulator, delta)
			centroids = candidates.collectAsMap()
			hasConverged = 0 == accumulator.value
			iteration += 1
		}
		printf("Converged at Iteration %s\n", iteration)
		// Convert the results back into an RDD so we can write them to HDFS.
		sc.parallelize(centroids.map { case (k,v) =>
			k + centroidDelim + v.mkString(pointDelim)
		}.toSeq).saveAsTextFile(cOut)
	}

	/**
	 * Parse the arguments into a suitable set of parameters to run the algorithm.
	 */
	def main(args:Array[String]) {
		if (args.length != 6) {
			Console.err.println("USAGE points centroids output max delta partitions")
			return
		}
		val conf = new SparkConf().setAppName("K-Means")
		sc = new SparkContext(conf)
		run(args(0).toString, // Points input.
			args(1).toString,   // Centroids input.
			args(2).toString,   // Centroids output.
			args(3).toInt,      // Maxium iterations.
			args(4).toDouble,   // Convergence delta.
			args(5).toInt)      // Partitions for points.
	}

}
