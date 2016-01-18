/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;

/*
 * An ensemble of different recommender models.
 */

import java.io.File;
import java.io.IOException;
import java.util.List;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class Recommenders {
	
	public static void main(String args[]) throws IOException, TasteException{
		
	DataModel model = new FileDataModel(new File("data/input/u1.base"));
	System.out.println(model.getNumItems());
	System.out.println(model.getNumUsers());
	System.out.println(model.getMaxPreference());
	System.out.println(model.getMinPreference());
	System.out.println(model.getPreferencesFromUser(1));
	
		Utilities utility = new Utilities();
		
		 /*User Based Recommender*/
		 System.err.println("User Based");
		 UserSimilarity userSimilarity = new PearsonCorrelationSimilarity (model);
		 UserNeighborhood neighborhood = new NearestNUserNeighborhood (10, userSimilarity, model);
		 Recommender userRecommender = new GenericUserBasedRecommender (model, neighborhood, userSimilarity);
		 List<RecommendedItem> userRecommendations = userRecommender.recommend(1, 3);
		 for (RecommendedItem recommendation : userRecommendations) {
			 	 utility.insert(new Long(recommendation.getItemID()),new Float(recommendation.getValue()));
		} 
		 utility.show();
		 
		 /*Item Based recommender*/
		 System.err.println("Item Based");
		 ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(model);
		 Recommender itemRecommender = new GenericItemBasedRecommender(model, itemSimilarity);
		 List<RecommendedItem> itemRecommendations = itemRecommender.recommend(1, 3);
		 for (RecommendedItem recommendation : itemRecommendations) {
			 	System.out.println(recommendation);
			 } 
		 
		 /*Slope One Recommender */
		 System.err.println("Slope Based");
		 SlopeOneRecommender slopeRecommender = new SlopeOneRecommender(model);
		 List<RecommendedItem> slopeRecommendations = slopeRecommender.recommend(1, 3);
		 for (RecommendedItem recommendation : slopeRecommendations) {
			 	System.out.println(recommendation);
			 }
		 
	}
}

