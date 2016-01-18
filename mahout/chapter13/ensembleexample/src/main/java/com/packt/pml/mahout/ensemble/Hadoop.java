/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;
/*
 * Implementing Mahout using Apache Hadoop in a distributed mode
 */
		
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.hadoop.item.RecommenderJob;

public class Hadoop {
	
	public static void main(String args[]) throws Exception{
		
/*	"--input /opt/hadoop/tmp/Mahout/rating1.csv --usersFile /opt/hadoop/tmp/Mahout/user1.txt " +
			"--similarityClassname SIMILARITY_COOCCURRENCE --tempDir /mahoutTemp --output /opt/hadoop/tmp/Mahout/output2.txt";*/

	Random r = new Random();
		
	String inputPath = "/opt/hadoop/tmp/Mahout/rating1.csv";
	String outputPath = "/opt/hadoop/tmp/Mahout/output6.txt";
	String tempPath = "/mahoutTemp/"+r.nextInt(Integer.MAX_VALUE);
	String userFile = "/opt/hadoop/tmp/Mahout/user2.txt";
	String Similarity = "SIMILARITY_COSINE";
	int numRecommendations = 1000;
	
	String home = "hdfs://ambo.hadoop.local:9000";
	String i = "--input" + " " + home+inputPath+ " "; 
	String o = "--output" + " " + home+outputPath+ " ";
	String t = "--tempDir" + " " + home+tempPath+ " ";
	String u = "--usersFile" + " " + home+userFile+ " ";
	String s = "--similarityClassname" + " " + Similarity+ " ";
	String n = "--numRecommendations" + " " + numRecommendations + " "; 
	
	String arguement = i + o + u + t + s + n;
	String [] argus = arguement.split(" ");
	
	
		for(String arg:argus)
			System.out.println(arg);
		
		
		Configuration conf = new Configuration();
		conf.set("fs.default.name", "hdfs://ambo.hadoop.local:9000");
		URI uri = new URI(conf.get("fs.default.name","."));
		try{
		FileSystem fs = FileSystem.get(uri,conf);
		
//		deleteTemp(fs,tempPath);
		
//		refreshTemp(fs, new Path(tempPath));
		
		RecommenderJob.main(argus);
		
		readRecom(fs,outputPath);
		}
		catch(Exception e){
			System.err.println(e.getLocalizedMessage());
		}
	}
	

	
	public static void readRecom(FileSystem fs1,String outputPath) throws URISyntaxException, IOException{

		String output = outputPath + "/part-r-00000";
		System.out.println(output);
				

        if(!fs1.exists(new Path(output)))
        	System.out.println("No file exists");
        
        BufferedReader br = new BufferedReader(new InputStreamReader(fs1.open(new Path(output))));
        String line = br.readLine();
        StringBuilder temp = new StringBuilder();
        while(line!=null){
        	temp.append(line + "\n");
        	line=br.readLine();
        }
        br.close();
		System.out.println(temp.toString());
	
		/*Path out = new Path(outputPath);
         if (fs1.exists(out)){
             fs1.delete(out, true);
             System.out.println("File deleted");
         }*/
	}
	
	public static void refreshTemp(FileSystem fs, Path tempPath) throws IOException{
		
		 if (fs.exists(tempPath)){
             fs.delete(tempPath, true);
             System.out.println("File deleted");
         }
		 
		 fs.create(tempPath);
		
	}
	

}
