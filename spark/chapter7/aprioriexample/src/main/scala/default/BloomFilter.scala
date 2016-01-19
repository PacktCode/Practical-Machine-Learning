// Practical Machine learning
// Association rule based learning - Apriori example
// Chapter 7

package default

import scala.collection.BitSet

class BloomFilter[T](
    buckets: Int,
    multiplier: Int,
    increment: Int,
    private val bitset: BitSet = BitSet()) extends Set[T] {
  import BloomFilter._

  override def contains(elem: T): Boolean =
      bitset.contains((reHash(multiplier, increment)(elem) % buckets))

  override def +(elem: T): Set[T] =
      new BloomFilter(
          buckets,
          multiplier,
          increment,
          bitset + (reHash(multiplier, increment)(elem) % buckets)
      )

  override def -(elem: T): Set[T] = ???

  override def iterator: Iterator[T] = ???
}

object BloomFilter extends App {
  def ??? : Nothing = throw new UnsupportedOperationException()

  def apply[T](buckets: Int, multiplier: Int = 12568, increment: Int = 76509)(elems: T*) = {
      new BloomFilter[T](
          buckets,
          multiplier,
          increment,
          BitSet(elems.map { elem: T => reHash(multiplier, increment)(elem) % buckets }: _*)
      )
  }

  def reHash(
      multiplier: Int,
      increment: Int
  )(a: Any): Int = a.hashCode() * multiplier + increment
}
