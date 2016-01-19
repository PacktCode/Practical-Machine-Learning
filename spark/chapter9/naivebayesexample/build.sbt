name := "blog-spark-naive-bayes-reuters"

version := "1.0"

scalaVersion := "2.10.4"

val sparkVersion = "1.0.0"

libraryDependencies <<= scalaVersion {
  scala_version => Seq(
    // Spark and Mllib
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    // Lucene
    "org.apache.lucene" % "lucene-core" % "4.8.1",
    // for Porter Stemmer
    "org.apache.lucene" % "lucene-analyzers-common" % "4.8.1",
    // Guava for the dictionary
    "com.google.guava" % "guava" % "17.0",
    // article extractor
    "com.gravity" %% "goose" % "2.1.23"
  )
}

// used for goose
resolvers += Resolver.mavenLocal
