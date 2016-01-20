package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import main.java.ComputeSimilarity;
import main.java.ExtractPatches;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class ExtractPatchesTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2161420777344271697L;
	private transient JavaSparkContext sc;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "FeatureExtractionTest");
	}
	
	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}
	
	@Test @Ignore
	public void extractPatchesTest() {
		
		// simple test example
		double[] x1 = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		double[] x2 = {0.86, 0.24, 1.24, 0.76, 0.98, 1.01, 2.1, 0.45, 0.81, 1.12, 0.45, 0.73, 0.23, 0.90, 0.80, 0.78};
		
		// create the parallel dataset
		List<Vector> matX = new ArrayList<Vector>(2);
		matX.add(Vectors.dense(x1));
		matX.add(Vectors.dense(x2));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		double[] expected_output = {0.56, 0.54, 0.34, 0.63,
									0.54, 1.23, 0.63, 0.34,
									1.23, 0.57, 0.34, 0.85,
									0.34, 0.63, 0.32, 1.2,
									0.63, 0.34, 1.2, 0.67,
									0.34, 0.85, 0.67, 0.29,
									0.32, 1.2, 0.14, 0.78,
									1.2, 0.67, 0.78, 0.85,
									0.67, 0.29, 0.85, 0.94,
									0.86, 0.24, 0.98, 1.01,
									0.24, 1.24, 1.01, 2.1,
									1.24, 0.76, 2.1, 0.45,
									0.98, 1.01, 0.81, 1.12,
									1.01, 2.1, 1.12, 0.45,
									2.1, 0.45, 0.45, 0.73,
									0.81, 1.12, 0.23, 0.9,
									1.12, 0.45, 0.9, 0.8,
									0.45, 0.73, 0.8, 0.78};
		int[] vecSize = {4,4};
		int[] patchSize = {2,2};
		
		// call the parallel extractPatches procedure
		JavaRDD<Vector> extractedPatches = matRDD.flatMap(new ExtractPatches(vecSize, patchSize));
		List<Vector> patchList = extractedPatches.collect();
		double[][] output = new double[18][4];
		
		// assign results to arrays
		for (int i = 0; i < patchList.size(); i++) {
			output[i] = patchList.get(i).toArray();
		}
		
		double[] outputD = new double[18*4];
		for (int i = 0; i < 18; i++) {
			for (int j = 0; j < 4; j++) {
				outputD[j+4*i] = output[i][j]; 
			}
		}
		
		Assert.assertArrayEquals(expected_output, outputD, 1e-6);
	}
	
}
