// Practical Machine learning
// Association rule based learning - Apriori example
// Chapter 7

package default

import scala.collection.SortedSet

trait FrequentItemSets {
  /**
   * Collection type for frequent item sets.
   *
   * @tparam T item type.
   */
  type ItemSet[T] = SortedSet[T]
}
