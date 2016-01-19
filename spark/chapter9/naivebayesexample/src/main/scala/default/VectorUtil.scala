// Practical Machine learning
// Bayesian learning - Naive Bayes example
// Chapter 9

package default

import com.google.common.collect.ImmutableBiMap
import scala.collection.JavaConversions._
import org.apache.spark.mllib.linalg.Vectors

class Dictionary(dict: Seq[String]) extends Serializable {

  // map term => index
  val termToIndex = ImmutableBiMap.builder[String, Int]()
    .putAll(dict.zipWithIndex.toMap[String, Int])
    .build()

  @transient
  lazy val indexToTerm = termToIndex.inverse()

  val count = termToIndex.size()

  def indexOf(term: String) = termToIndex(term)

  def valueOf(index: Int) = indexToTerm(index)

  def tfIdfs(terms: Seq[String], idfs: Map[String, Double]) = {
    val filteredTerms = terms.filter(idfs contains)
    (filteredTerms.groupBy(identity).map {
      case (term, instances) =>
        (indexOf(term), (instances.size.toDouble / filteredTerms.size.toDouble) * idfs(term))
    }).toSeq.sortBy(_._1) // sort by termId
  }

  def vectorize(tfIdfs: Iterable[(Int, Double)]) = {
    Vectors.sparse(dict.size, tfIdfs.toSeq)
  }
}
