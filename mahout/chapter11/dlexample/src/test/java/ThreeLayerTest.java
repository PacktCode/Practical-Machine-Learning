package test.java;

import static org.junit.Assert.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;
import main.java.BaseLayerFactory;
import main.java.DeepLearningLayer;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigKMeans;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DeepModelSettings.ConfigPreprocess;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class ThreeLayerTest implements Serializable {

	private static final long serialVersionUID = -597804824623690421L;
	private transient JavaSparkContext sc;
	ConfigBaseLayer config1;
	ConfigBaseLayer config2;
	ConfigBaseLayer config3;
	
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "ThreeLayerTest");
		// Configuration for layer1 
		ConfigBaseLayer.Builder conf = ConfigBaseLayer.newBuilder();
		conf.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf.setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(2).setFeatureDim2(2).setInputDim1(32).setInputDim2(32).build());
		conf.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(3).setNumberOfIterations(10).build());	
	 	config1 = conf.build();
	 	
		// Configuration for layer2 
		ConfigBaseLayer.Builder conf2 = ConfigBaseLayer.newBuilder();
		conf2.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf2.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf2.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(4).setNumberOfIterations(10).build());	
	 	config2 = conf2.build();
	 	
		// Configuration for layer3 
		ConfigBaseLayer.Builder conf3 = ConfigBaseLayer.newBuilder();
		conf3.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf3.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf3.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(11).setNumberOfIterations(10).build());	
	 	config3 = conf3.build();
	}

	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}
	
	@Test @Ignore
	public void testSmallLoop() throws Exception {

	 	List<ConfigBaseLayer> config_list = new ArrayList<ConfigBaseLayer>();
	 	config_list.add(config1);
	 	config_list.add(config2);
	 	config_list.add(config3);
	 	
	 	int Nimgs = 50;
	 	int Npatches = 100;
	 	List<Vector> input_word_patches = new ArrayList<Vector>(Nimgs);
	 	int S = 32;
	 	double[] temp = new double[S*S];
 		for (int j = 0; j < S*S; ++j) {
 			temp[j] = (double)j;
 		}
	 	for (int i = 0; i < Nimgs; ++i) {
	 		input_word_patches.add(Vectors.dense(temp));
	 	}
	 	
	 	List<Vector> input_small_patches = new ArrayList<Vector>(Npatches);
	 	for (int i = 0; i < Npatches; ++i) {
	 		input_small_patches.add(Vectors.dense(1,2,3,4));
	 	}
		// We have 100 patches of size 2x2 as input
		// We have 50 word images of size 8x8
		JavaRDD<Vector> patches = sc.parallelize(input_small_patches);
		JavaRDD<Vector> imgwords = sc.parallelize(input_word_patches);
		JavaRDD<Vector> result = null;
		int layer_index = 0;
	 	for (ConfigBaseLayer config_layer: config_list) {
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(config_layer, layer_index++, "three_layer");
			// The config layer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (config_layer.hasConfigFeatureExtractor()) {
				result = layer.train(patches, imgwords,false);
			} else {
				result = layer.train(result, result,false);
			}	
	 	}
		List<Vector> out = result.collect();
		Assert.assertEquals(50, out.size());
		Assert.assertEquals(5, out.get(0).size());	
	}
}