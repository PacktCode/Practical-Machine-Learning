/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;
/*
 * Evaluate a Recommendation Model
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;

class Builder implements RecommenderBuilder{
	int k;
	UserSimilarity similarity;
	DataModel dataModel;
	
	Builder(int k, int similarityMeasure, DataModel dataModel) throws TasteException{
		this.k=k;
		this.dataModel=dataModel;
		if(similarityMeasure==0)
			similarity = new EuclideanDistanceSimilarity(dataModel);
		else
			similarity = new PearsonCorrelationSimilarity(dataModel);
			
	}
	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		UserNeighborhood neighbors = new NearestNUserNeighborhood(k, similarity, dataModel);
		return new GenericUserBasedRecommender(dataModel, neighbors, similarity);
	}
	
}

public class RecommenderEvaluator {
	
	public static void main(String args[]) throws IOException, TasteException{
		String result = evaluateOnSimilarity(200,0);
		System.out.println(result);
		fileWrite(new File("Euc_User.txt"), result);
	}

	public static String evaluateOnSimilarity(int k,int SimilarityMeasure) throws IOException, TasteException{
		
		DataModel model = new FileDataModel(new File("data/input/u.data"));
		RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		Builder builder; 
		double score;
		StringBuilder result = new StringBuilder();
		for(int i =1; i<=k;){
			builder = new Builder(i,0,model);
			score = evaluator.evaluate(builder, null, model, 0.8, 0.7);
			result.append(i + "\t" + score + System.lineSeparator());
			System.out.println(i);
			i=i+5;
		}
		return result.toString();
		
	}
	
	public static void fileWrite(File file,String text) throws IOException{
		BufferedWriter bw = new BufferedWriter(new FileWriter(file));
		
		bw.write(text);
		bw.flush();
		bw.close();
	}
}
