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

public class OneLayerTest implements Serializable {

	private static final long serialVersionUID = -8953031572106667936L;
	private transient JavaSparkContext sc;

	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "OneLayerTest");
	}

	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}

	@Test @Ignore
	public void test() throws Exception {
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
	 	
		DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(c, 0, "one_layer");
		// We have 100 patches of size 2x2 as input
		// We have 50 word images of size 8x8
		JavaRDD<Vector> patches = sc.parallelize(input_small_patches);
		JavaRDD<Vector> imgwords = sc.parallelize(input_word_patches);
		
	   	JavaRDD<Vector> preprocessed = layer.preProcess(patches);
	   	List<Vector> v = preprocessed.collect();
	   	Assert.assertEquals(100, v.size());
		Vector[] features = layer.learnFeatures(preprocessed);
		Assert.assertEquals(3, features.length);
		Assert.assertEquals(4, features[0].size());
		
		JavaRDD<Vector> represent = layer.extractFeatures(imgwords, features);
		
		List<Vector> t = represent.collect();
		Assert.assertEquals(50, t.size());
		// 147 is 3 x 7 x 7
		Assert.assertEquals(147, t.get(0).size());
		JavaRDD<Vector> pooled = layer.pool(represent);
		List<Vector> result = pooled.collect();
		Assert.assertEquals(50, result.size());
		Assert.assertEquals(27, result.get(0).size());

	}
	
	@Test @Ignore
	public void testBaseLayer() throws Exception {
		ConfigBaseLayer.Builder conf = ConfigBaseLayer.newBuilder();
		conf.setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.2).build());
		conf.setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(2).setFeatureDim2(2).setInputDim1(8).setInputDim2(8).build());
		conf.setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2));
	 	conf.setConfigKmeans(ConfigKMeans.newBuilder().setNumberOfClusters(3).setNumberOfIterations(10).build());	
	 	ConfigBaseLayer layer_config = conf.build();
	 	
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
	 	
	 	int layer_index = 1;
		DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(layer_config, layer_index, "one_layer_2");
		layer.setSparkContext(sc);
		// We have 100 patches of size 2x2 as input
		// We have 50 word images of size 8x8
		JavaRDD<Vector> patches = sc.parallelize(input_small_patches);
		JavaRDD<Vector> imgwords = sc.parallelize(input_word_patches);
		
		JavaRDD<Vector> result = layer.train(patches, imgwords,false);

		List<Vector> res = result.collect();
		Assert.assertEquals(50, res.size());
		Assert.assertEquals(27, res.get(0).size());

	}
}

