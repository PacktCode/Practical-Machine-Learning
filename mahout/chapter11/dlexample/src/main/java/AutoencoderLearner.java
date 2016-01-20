// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigKMeans;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public class AutoencoderLearner implements Learner{
	
	private AutoencoderConfig conf;
	
	public AutoencoderLearner(ConfigBaseLayer configLayer) {
		this.conf = new AutoencoderConfig(configLayer);
	}

	@Override
	public Vector[] call(JavaRDD<Vector> data) throws Exception {
		return new Autoencoder(conf).train(data);
	}

}
