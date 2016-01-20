/**
 * 
 */
package test.java;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.MaxPooler;
import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author Viviana Petrescu
 *
 */
public class MaxPoolerTest implements Serializable{
	private static final long serialVersionUID = 145346357547456L;
	private transient JavaSparkContext sc;
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "MaxPoolerTest");
	}

	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}

	@Test
	public void sampleTest() {
		//fail("Not yet implemented");
		List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
		JavaRDD<Integer> distData = sc.parallelize(data);
		class Sum implements Function2<Integer, Integer, Integer>, Serializable {
			private static final long serialVersionUID = 2685928850298905497L;

			public Integer call(Integer a, Integer b) { return a + b; }
		}

		int totalLength = distData.reduce(new Sum());
		Assert.assertEquals(15, totalLength);
	}
	@Test
	public void test1DMaxPooling() {
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2)).build();
		MaxPooler pooler = new MaxPooler(conf);
		double[] input = {1,2,3,4,5,6,7,8,9,10};
		Vector data = Vectors.dense(input);
		Vector output = pooler.call(data); //poolOver1D
		Assert.assertEquals(5, output.size());
		double[] expected_outputs = {2,4,6,8,10};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	@Test
	public void test2DMaxPooling() {
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(1).setFeatureDim2(1).setInputDim1(6).setInputDim2(4)).
				setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2)).build();
		MaxPooler pooler = new MaxPooler(conf);
		Assert.assertEquals(true, pooler.isPoolOver2DInput());
		// The 6x4 matrix shown below is stored as a 1D vector in variable input.
		/* 1 7  13 19
		 * 2 8  14 20
		 * 3 9  15 21
		 * 4 10 16 22
		 * 5 11 17 23
		 * 6 12 18 24
		  */
		double[] input = {1,  2,  3,  4,  5,  6,
				          7,  8,  9,  10, 11, 12,
				          13, 14, 15, 16, 17, 18,
				          19, 20, 21, 22, 23, 24};

		Vector data = Vectors.dense(input);
		// Check that the input dimensions were correctly computed.
		Assert.assertEquals(6, pooler.getInputDim1());
		Assert.assertEquals(4, pooler.getInputDim2());
		Assert.assertEquals(2, conf.getConfigPooler().getPoolSize());
		Vector output = pooler.call(data); // poolOver2D
		Assert.assertEquals(6, output.size());
		double[] expected_outputs = {8,10,12,20,22,24};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	//TODO add MaxPoolerExtended test
}
