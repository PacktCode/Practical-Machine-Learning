/**
* Practical Machine learning
* Clustering based learning - K-means clustering Example
* Chapter 8
*/
package com.packt.pml.mahout.kmeans;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;

import chapter7.src.InputDriver;

public class DataPreprocessing {
	
	public static void main(String args[]) throws ClassNotFoundException, IOException, InterruptedException
	{
	
	Configuration conf = new Configuration();
	conf.addResource(new Path("/usr/local/hadoop/conf/core-site.xml"));
	conf.addResource(new Path("/usr/local/hadoop/conf/hdfs-site.xml"));

	//create the file system object and pass the configuration object		
	FileSystem fileSystem = FileSystem.get(conf);
	//We then create the input and output Path Objects.

			
	//define the input and sequence file directory
	String inputPath="chapter7/clustering_input";
	String inputSeq="clustering_seq";
			
	Path inputDir = new Path(inputPath);
	Path inputSeqDir = new Path(inputSeq);
	
    if (fileSystem.exists(inputSeqDir)) {
		System.out.println("Output already exists");
		fileSystem.delete(inputSeqDir, true);
		System.out.println("deleted output directory");
	}

	//The last step is to encode the vectors using the //RandomAccessSparseVector
	InputDriver.runJob(inputDir, inputSeqDir, 
			"org.apache.mahout.math.RandomAccessSparseVector",conf);

	}
}
