package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import main.java.ComputeSimilarity;
import main.java.ExtractPatches;
import main.java.ExtractPatchesTuples;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import scala.Tuple2;

public class ExtractPatchesTuplesTest implements Serializable {

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
	
	@Test
	public void extractPatchesTest() {
		
		// simple test example
		double[] x1 = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		double[] x2 = {0.86, 0.24, 1.24, 0.76, 0.98, 1.01, 2.1, 0.45, 0.81, 1.12, 0.45, 0.73, 0.23, 0.90, 0.80, 0.78};
		
		double[] m1 = {32, 42};
		double[] m2 = {21, 25};
		
		// create the parallel dataset
		List<Tuple2<Vector, Vector>> matX = new ArrayList<Tuple2<Vector, Vector>>(2);
		matX.add(new Tuple2<Vector, Vector>(Vectors.dense(m1), Vectors.dense(x1)));
		matX.add(new Tuple2<Vector, Vector>(Vectors.dense(m2), Vectors.dense(x2)));
		JavaRDD<Tuple2<Vector, Vector>> matRDD = sc.parallelize(matX);
		
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
		JavaRDD<Tuple2<Vector, Vector>> extractedPatches = matRDD.flatMap(new ExtractPatchesTuples(vecSize, patchSize));
		List<Tuple2<Vector, Vector>> patchList = extractedPatches.collect();
		double[][] output = new double[18][4];
		
		// assign results to arrays
		for (int i = 0; i < patchList.size(); i++) {
			output[i] = patchList.get(i)._2.toArray();
			
			// print results
			System.out.println("Vector #1 of the tuple2 #" + i);
			System.out.println(patchList.get(i)._1);
			
			System.out.println("Vector #2 of the tuple2 #" + i);
			System.out.println(patchList.get(i)._2);
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
