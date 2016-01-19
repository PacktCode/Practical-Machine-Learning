sbt package
rm -rf final
spark-submit \
  --class "com.jgalilee.spark.kmeans.JobDriver" \
  --master local[4] \
  ./target/scala-2.10/spark-k-means_2.10-1.0.jar \
  input/points.txt input/centroids.txt final 10 0.0 3
cat final/p* | sort
