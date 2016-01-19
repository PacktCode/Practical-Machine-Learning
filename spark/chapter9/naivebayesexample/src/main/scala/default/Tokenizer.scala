// Practical Machine learning
// Bayesian learning - Naive Bayes example
// Chapter 9

package default


import java.io.StringReader
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.util.Version
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import scala.collection.mutable

object Tokenizer {
  val LuceneVersion = Version.LUCENE_48

  def tokenizeAll(docs: Iterable[Document]) = docs.map(tokenize)

  def tokenize(doc: Document): TermDoc = TermDoc(doc.docId, doc.labels, tokenize(doc.body))

  def tokenize(content: String): Seq[String] = {
    val tReader = new StringReader(content)
    val analyzer = new EnglishAnalyzer(LuceneVersion)
    val tStream = analyzer.tokenStream("contents", tReader)
    val term = tStream.addAttribute(classOf[CharTermAttribute])
    tStream.reset()

    val result = mutable.ArrayBuffer.empty[String]
    while(tStream.incrementToken()) {
      val termValue = term.toString
      if (!(termValue matches ".*[\\d\\.].*")) {
        result += term.toString
      }
    }
    result
  }
}

case class TermDoc(doc: String, labels: Set[String], terms: Seq[String])