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

public class TwoLayersTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6070301047615364301L;
	private transient JavaSparkContext sc;
	ConfigBaseLayer config1;
	ConfigBaseLayer config2;

	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "TwoLayerTest");
		// Configuration for layer1 
		ConfigBaseLayer.Builder conf = ConfigBaseLayer.newBuilder();
		conf.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf.setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(2).setFeatureDim2(2).setInputDim1(8).setInputDim2(8).build());
		conf.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(3).setNumberOfIterations(10).build());	
	 	config1 = conf.build();
	 	
		// Configuration for layer2 
		ConfigBaseLayer.Builder conf2 = ConfigBaseLayer.newBuilder();
		conf2.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf2.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf2.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(4).setNumberOfIterations(10).build());	
	 	config2 = conf2.build();
	}

	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}

	@Test @Ignore
	public void test() throws Exception {
	 	
	 	int Nimgs = 50;
	 	int Npatches = 100;
	 	List<Vector> input_word_patches = new ArrayList<Vector>(Nimgs);
	 	double[] temp = new double[64];
 		for (int j = 0; j < 64; ++j) {
 			temp[j] = (double)j;
 		}
	 	for (int i = 0; i < Nimgs; ++i) {
	 		input_word_patches.add(Vectors.dense(temp));
	 	}
	 	
	 	List<Vector> input_small_patches = new ArrayList<Vector>(Npatches);
	 	for (int i = 0; i < Npatches; ++i) {
	 		input_small_patches.add(Vectors.dense(1,2,3,4));
	 	}
	 	
		DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(config1, 0, "two_layer1");
		// We have 100 patches of size 2x2 as input
		// We have 50 word images of size 8x8
		JavaRDD<Vector> patches = sc.parallelize(input_small_patches);
		JavaRDD<Vector> imgwords = sc.parallelize(input_word_patches);
		
		JavaRDD<Vector> result = layer.train(patches, imgwords,false);

		List<Vector> res = result.collect();
		Assert.assertEquals(50, res.size());
		Assert.assertEquals(27, res.get(0).size());

		
	 	DeepLearningLayer layer2 = BaseLayerFactory.createBaseLayer(config2, 1,"two_layer1");
	 	
    	JavaRDD<Vector> preprocessed = layer2.preProcess(result);
		Vector[] features = layer2.learnFeatures(preprocessed);
       
		Assert.assertEquals(50, imgwords.collect().size());
		Assert.assertEquals(4, features.length);
		Assert.assertEquals(27, features[0].size());
		Assert.assertEquals(64,imgwords.collect().get(0).size());
		JavaRDD<Vector> represent = layer2.extractFeatures(result, features);
		Assert.assertEquals(50, represent.collect().size());
		Assert.assertEquals(4, represent.collect().get(0).size());
		
		JavaRDD<Vector> pooled = layer2.pool(represent);
		List<Vector> out = pooled.collect();
		Assert.assertEquals(50, out.size());
		Assert.assertEquals(2, out.get(0).size());
	 	
	}
	
	@Test @Ignore
	public void testForLoop() throws Exception {
		// Configuration for layer1 
		ConfigBaseLayer.Builder conf = ConfigBaseLayer.newBuilder();
		conf.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf.setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(2).setFeatureDim2(2).setInputDim1(8).setInputDim2(8).build());
		conf.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(3).setNumberOfIterations(10).build());	
	 	ConfigBaseLayer c = conf.build();
	 	
	 	int Nimgs = 50;
	 	int Npatches = 100;
	 	List<Vector> input_word_patches = new ArrayList<Vector>(Nimgs);
	 	double[] temp = new double[64];
 		for (int j = 0; j < 64; ++j) {
 			temp[j] = (double)j;
 		}
	 	for (int i = 0; i < Nimgs; ++i) {
	 		input_word_patches.add(Vectors.dense(temp));
	 	}
	 	
	 	List<Vector> input_small_patches = new ArrayList<Vector>(Npatches);
	 	for (int i = 0; i < Npatches; ++i) {
	 		input_small_patches.add(Vectors.dense(1,2,3,4));
	 	}
	 	int layer_index = 0;
		DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(c, layer_index, "two_layer2");
		// We have 100 patches of size 2x2 as input
		// We have 50 word images of size 8x8
		JavaRDD<Vector> patches = sc.parallelize(input_small_patches);
		JavaRDD<Vector> imgwords = sc.parallelize(input_word_patches);
		
		JavaRDD<Vector> result = layer.train(patches, imgwords,false);

		List<Vector> res = result.collect();
		Assert.assertEquals(50, res.size());
		Assert.assertEquals(27, res.get(0).size());
		
		// Configuration for layer2 
		ConfigBaseLayer.Builder conf2 = ConfigBaseLayer.newBuilder();
		conf2.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf2.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf2.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(4).setNumberOfIterations(10).build());	
	 	ConfigBaseLayer c2 = conf2.build();
		
	 	layer_index = 1;
	 	DeepLearningLayer layer2 = BaseLayerFactory.createBaseLayer(c2, layer_index, "two_layer_2");
		JavaRDD<Vector> result2 = layer2.train(result, result,false);

		List<Vector> out = result2.collect();
		Assert.assertEquals(50, out.size());
		Assert.assertEquals(2, out.get(0).size());	
	}
	@Test @Ignore
	public void testSmallLoop() throws Exception {

	 	List<ConfigBaseLayer> config_list = new ArrayList<ConfigBaseLayer>();
	 	config_list.add(config1);
	 	config_list.add(config2);
	 	
	 	int Nimgs = 50;
	 	int Npatches = 100;
	 	List<Vector> input_word_patches = new ArrayList<Vector>(Nimgs);
	 	double[] temp = new double[64];
 		for (int j = 0; j < 64; ++j) {
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
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(config_layer, layer_index++, "two_layer3");
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
		Assert.assertEquals(2, out.get(0).size());	
	}
}
