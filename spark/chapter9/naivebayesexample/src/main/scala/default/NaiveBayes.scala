// Practical Machine learning
// Bayesian learning - Naive Bayes example
// Chapter 9

package default


import java.io.{File, FilenameFilter}

import com.gravity.goose.{Configuration, Goose}
import jline.ConsoleReader
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object NaiveBayesExample extends App {

  // 4 workers
  val sc = new SparkContext("local[4]", "naivebayes")

  val dataDir = if (args.length == 1) {
    args.head
  } else {
    "reuters"
  }

  val naiveBayesAndDictionaries = createNaiveBayesModel(dataDir)
  console(naiveBayesAndDictionaries)

  /**
   * REPL loop to enter different urls
   */
  def console(naiveBayesAndDictionaries: NaiveBayesAndDictionaries) = {
    println("Enter 'q' to quit")
    val consoleReader = new ConsoleReader()
    while ( {
      consoleReader.readLine("url> ") match {
        case s if s == "q" => false
        case url: String =>
          predict(naiveBayesAndDictionaries, url)
          true
        case _ => true
      }
    }) {}

    sc.stop()
  }

  def predict(naiveBayesAndDictionaries: NaiveBayesAndDictionaries, url: String) = {
    // extract content from web page
    val config = new Configuration
    config.setEnableImageFetching(false)
    val goose = new Goose(config)
    val content = goose.extractContent(url).cleanedArticleText
    // tokenize content and stem it
    val tokens = Tokenizer.tokenize(content)
    // compute TFIDF vector
    val tfIdfs = naiveBayesAndDictionaries.termDictionary.tfIdfs(tokens, naiveBayesAndDictionaries.idfs)
    val vector = naiveBayesAndDictionaries.termDictionary.vectorize(tfIdfs)
    val labelId = naiveBayesAndDictionaries.model.predict(vector)

    // convert label from double
    println("Label: " + naiveBayesAndDictionaries.labelDictionary.valueOf(labelId.toInt))
  }

  /**
   *
   * @param directory
   * @return
   */
  def createNaiveBayesModel(directory: String) = {
    val inputFiles = new File(directory).list(new FilenameFilter {
      override def accept(dir: File, name: String) = name.endsWith(".sgm")
    })

    val fullFileNames = inputFiles.map(directory + "/" + _)
    val docs = ReutersParser.parseAll(fullFileNames)
    val termDocs = Tokenizer.tokenizeAll(docs)

    // put collection in Spark
    val termDocsRdd = sc.parallelize[TermDoc](termDocs.toSeq)

    val numDocs = termDocs.size

    // create dictionary term => id
    // and id => term
    val terms = termDocsRdd.flatMap(_.terms).distinct().collect().sortBy(identity)
    val termDict = new Dictionary(terms)

    val labels = termDocsRdd.flatMap(_.labels).distinct().collect()
    val labelDict = new Dictionary(labels)

    // compute TFIDF and generate vectors
    // for IDF
    val idfs = (termDocsRdd.flatMap(termDoc => termDoc.terms.map((termDoc.doc, _))).distinct().groupBy(_._2) collect {
      // mapValues not implemented :-(
      // if term is present in less than 3 documents then remove it
      case (term, docs) if docs.size > 3 =>
        term -> (numDocs.toDouble / docs.size.toDouble)
    }).collect.toMap

    val tfidfs = termDocsRdd flatMap {
      termDoc =>
        val termPairs = termDict.tfIdfs(termDoc.terms, idfs)
        // we consider here that a document only belongs to the first label
        termDoc.labels.headOption.map {
          label =>
            val labelId = labelDict.indexOf(label).toDouble
            val vector = Vectors.sparse(termDict.count, termPairs)
            LabeledPoint(labelId, vector)
        }
    }

    val model = NaiveBayes.train(tfidfs)
    NaiveBayesAndDictionaries(model, termDict, labelDict, idfs)
  }
}

case class NaiveBayesAndDictionaries(model: NaiveBayesModel,
                                     termDictionary: Dictionary,
                                     labelDictionary: Dictionary,
                                     idfs: Map[String, Double])
