name := "NeuralNetwork"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.2.1",
  "org.apache.spark" % "spark-mllib_2.10" % "1.2.1",
  "org.scalanlp" % "breeze_2.10" % "0.10",
  "org.scalanlp" % "breeze-natives_2.10" % "0.10",
  "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test"
)