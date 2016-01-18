/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;

/*
 * A item based recommender model.
 */

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;

class ItembasedBuilder implements RecommenderBuilder{
	int k;
	ItemSimilarity similarity;
	
	ItembasedBuilder(int similarityMeasure, DataModel dataModel) throws TasteException{
		
		if(similarityMeasure==0)
			similarity = new EuclideanDistanceSimilarity(dataModel);
		else
			similarity = new PearsonCorrelationSimilarity(dataModel);
	}
	public Recommender buildRecommender(DataModel dataModel) throws TasteException {
		return new GenericItemBasedRecommender(dataModel, similarity);
	}
	
}

public class ItemRecommender {
	
	public static void main(String args[]) throws IOException, TasteException{
		
		DataModel model = new FileDataModel(new File("data/input/u1.base"));
		RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		ItembasedBuilder builder; 
		double score;
		builder = new ItembasedBuilder(1,model);
		score = evaluator.evaluate(builder, null, model, 0.8, 0.7);
		System.out.println(score);
   }
}