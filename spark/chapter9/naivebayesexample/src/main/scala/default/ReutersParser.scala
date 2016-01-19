// Practical Machine learning
// Bayesian learning - Naive Bayes example
// Chapter 9

package default


import scala.xml.pull.{EvText, EvElemEnd, EvElemStart, XMLEventReader}
import scala.io.Source
import scala.collection.mutable

object ReutersParser {
  def PopularCategories = Seq("money", "fx", "crude", "grain", "trade", "interest", "wheat", "ship", "corn", "oil", "dlr", "gas", "oilseed", "supply", "sugar", "gnp", "coffee", "veg", "gold", "nat", "soybean", "bop", "livestock", "cpi")

  def parseAll(xmlFiles: Iterable[String]) = xmlFiles flatMap parse

  def parse(xmlFile: String) = {
    val docs = mutable.ArrayBuffer.empty[Document]
    val xml = new XMLEventReader(Source.fromFile(xmlFile, "latin1"))
    var currentDoc: Document = null
    var inTopics = false
    var inLabel = false
    var inBody = false
    for (event <- xml) {
      event match {
        case EvElemStart(_, "REUTERS", attrs, _) =>
          currentDoc = Document(attrs.get("NEWID").get.head.text)

        case EvElemEnd(_, "REUTERS") =>
          if (currentDoc.labels.nonEmpty) {
            docs += currentDoc
          }

        case EvElemStart(_, "TOPICS", _, _) => inTopics = true

        case EvElemEnd(_, "TOPICS") => inTopics = false

        case EvElemStart(_, "D", _, _) => inLabel = true

        case EvElemEnd(_, "D") => inLabel = false

        case EvElemStart(_, "BODY", _, _) => inBody = true

        case EvElemEnd(_, "BODY") => inBody = false

        case EvText(text) =>
          if (text.trim.nonEmpty) {
            if (inTopics && inLabel && PopularCategories.contains(text)) {
              currentDoc = currentDoc.copy(labels = currentDoc.labels + text)
            } else if (inBody) {
              currentDoc = currentDoc.copy(body = currentDoc.body + text.trim)
            }
          }

        case _ =>
      }
    }
    docs
  }
}

case class Document(docId: String, body: String = "", labels: Set[String] = Set.empty)
