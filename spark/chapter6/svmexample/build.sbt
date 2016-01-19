name := "Spark Kernel SVM"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies ++=  Seq(
"org.apache.spark" %% "spark-core" % "1.3.1",
"org.apache.spark"  %% "spark-mllib"             % "1.3.1",
"amplab" % "spark-indexedrdd" % "0.1"
)
