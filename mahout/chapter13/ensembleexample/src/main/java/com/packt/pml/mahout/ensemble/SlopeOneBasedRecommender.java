/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;

/*
 * A slope one based recommender model.
 */

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;


public class SlopeOneBasedRecommender {
	
	public static void main(String args[]) throws IOException, TasteException{
		
		DataModel model = new FileDataModel(new File("data/input/u.data"));
		RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
		
		RecommenderBuilder builder = new RecommenderBuilder() {
		 public Recommender buildRecommender(DataModel model)throws TasteException { 
			 SlopeOneRecommender slope = new SlopeOneRecommender(model);
			 System.out.println(slope.recommend(199, 3).toString());
		 
			 return slope;
			}
		};
		
		double score = evaluator.evaluate(builder, null, model, 0.8, 0.7);
		System.out.println(score);
   }
}

/*OutPut
 * Score = 0.9507197266125407
 * [RecommendedItem[item:1175, value:7.0], RecommendedItem[item:1158, value:6.0], RecommendedItem[item:1026, value:5.7245636]]
 */