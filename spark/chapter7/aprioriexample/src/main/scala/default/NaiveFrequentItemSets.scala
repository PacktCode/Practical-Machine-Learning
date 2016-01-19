// Practical Machine learning
// Association rule based learning - Apriori example
// Chapter 7

package default

import scala.collection.Map
import scala.reflect.ClassTag

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

/**
 * Utility to compute frequent item sets for association rule mining.
 *
 * @see [[http://en.wikipedia.org/wiki/Association_rule_learning]]
 */
object NaiveFrequentItemSets extends FrequentItemSets {
  /**
   * Computes frequent item sets up to the specified size.
   *
   * @param sparkContext context in which to run. This is used to broadcast shared memory.
   * @param baskets to analyze.
   * @param supportThreshold minimum # of times an item set must occur to be considered frequent.
   * @param maxSize maximum item set size.
   * @tparam T item type.
   * @return map of frequent item sets and their counts.
   */
  def apply[T: Ordering: ClassTag](sparkContext: SparkContext)(
      baskets: RDD[ItemSet[T]],
      supportThreshold: Int,
      maxSize: Int): Map[ItemSet[T], Int] = {
    // Count item subsets from size 1 up to maxSize.
    baskets.flatMap { basket: ItemSet[T] =>
      (1 to maxSize).map(basket.subsets(_)).reduce(_ ++ _).map((_, 1))
    }
    .reduceByKey(_ + _)
    // Filter by support threshold.
    .filter { case (itemSet: ItemSet[T], count: Int) => count >= supportThreshold }
    .collectAsMap()
  }
}
