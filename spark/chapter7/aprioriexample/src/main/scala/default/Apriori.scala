// Practical Machine learning
// Association rule based learning - Apriori example
// Chapter 7

package default

import scala.annotation.tailrec
import scala.collection.Map
import scala.collection.Set
import scala.reflect.ClassTag

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

/**
 * Utility to compute frequent item sets for association rule mining. Uses A Priori algorithm.
 *
 * @see [[http://en.wikipedia.org/wiki/Association_rule_learning]]
 */
object APriori extends FrequentItemSets {
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
    getFrequentNSets[T](sparkContext)(
        baskets,
        supportThreshold,
        maxSize,
        1, // Start with item sets of size 1.
        sparkContext.broadcast(Map[Int, Map[ItemSet[T], Int]]())
    )
  }

  /**
   * Recursively computes frequent item sets from specified size up to specified maximum size.
   *
   * @param sparkContext context in which to run. This is used to broadcast shared memory.
   * @param baskets to analyze.
   * @param supportThreshold minimum # of times an item set must occur to be considered frequent.
   * @param maxSize maximum item set size.
   * @param size size of frequent item sets to compute.
   * @param collectedResults aggregate frequent itemset results in shared memory keyed by set size.
   * @tparam T item type.
   * @return map of frequent item sets and their counts.
   */
  @tailrec private def getFrequentNSets[T: Ordering: ClassTag](sparkContext: SparkContext)(
      baskets: RDD[ItemSet[T]],
      supportThreshold: Int,
      maxSize: Int,
      size: Int,
      collectedResults: Broadcast[Map[Int, Map[ItemSet[T], Int]]]): Map[ItemSet[T], Int] = {
    // Terminate recursion if size exceed maxSize.
    if (maxSize < size) {
      return collectedResults.value.values.reduce(_ ++ _)
    } else {
      // Compute item sets of specified size.
      val frequentNSets: Map[ItemSet[T], Int] = getFrequentNSets(
          size,
          baskets,
          if (size == 1) None else Some(collectedResults.value.get(size - 1).get.keySet),
          supportThreshold
      )

      // Append to collected results.
      val newCollectedResults: Map[Int, Map[ItemSet[T], Int]] =
          collectedResults.value + (size -> frequentNSets)

      // Clean up results in shared memory.
      collectedResults.destroy()

      // Compute item sets of next size.
      return getFrequentNSets(sparkContext)(
          baskets,
          supportThreshold,
          maxSize,
          size + 1,
          sparkContext.broadcast(newCollectedResults)
      )
    }
  }

  /**
   * Get frequent item sets of specified size given frequent item sets of size - 1.
   *
   * @param size size of frequent item sets to compute.
   * @param baskets to analyze.
   * @param frequentNMinus1Sets frequent item sets of size - 1.
   * @param supportThreshold minimum # of times an item set must occur to be considered frequent.
   * @tparam T item type.
   * @return map of frequent item sets and their counts.
   */
  private def getFrequentNSets[T: ClassTag](
      size: Int,
      baskets: RDD[ItemSet[T]],
      frequentNMinus1Sets: Option[Set[ItemSet[T]]],
      supportThreshold: Int): Map[ItemSet[T], Int] = {
    // Expand baskets.
    baskets.flatMap { basket: ItemSet[T] =>
      val subBaskets: Iterator[ItemSet[T]] = basket.subsets(size)
      frequentNMinus1Sets match {
        case Some(_) => {
          // Filter if frequent item sets of size - 1 were provided.
          subBaskets.filter { candidate: ItemSet[T] =>
            candidate.subsets(size - 1).forall(frequentNMinus1Sets.get.contains(_))
          }
        }
        case None => subBaskets
      }
    }
    // Count item sets.
    .map((_, 1))
    .reduceByKey(_ + _)
    // Filter by support threshold.
    .filter { case (itemSet: ItemSet[T], count: Int) => count >= supportThreshold }
    .collectAsMap()
  }
}

