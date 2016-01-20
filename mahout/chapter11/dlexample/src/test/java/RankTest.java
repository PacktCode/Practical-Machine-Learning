package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import main.java.ComputeSimilarity;
import main.java.MatrixOps;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class RankTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8707243339400493968L;
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
	public void rankTest() {
		
		// simple example
		double[] x1 = {0.35, 0.65, 0.28, 0.12}; 
		double[] x2 = {0.86, 0.96, 0.34, 0.57};
		double[] query = {0.46, 0.92, 0.78, 0.34};
		
		double[] expected_output = {0.955073918586867, 0.897967422096528};
		
		Vector queryV = Vectors.dense(query);
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(2);
		matX.add(Vectors.dense(x1));
		matX.add(Vectors.dense(x2));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// compute cosine similarities
		JavaRDD<Double> sims = matRDD.map(new ComputeSimilarity(queryV));
		
		final Double[] output = sims.collect().toArray(new Double[2]);
		final double[] outputD = ArrayUtils.toPrimitive(output);
		
		// sort the similarities and the indices
		final Integer[] idx = new Integer[2];
		for (int i = 0; i < 2; i++) {
			idx[i] = i;
		}
		Arrays.sort(idx, new Comparator<Integer>() {
		    @Override 
		    public int compare(final Integer o1, final Integer o2) {
		        return Double.compare(outputD[o1], outputD[o2]);
		    }
		});
		System.out.println("Sorted indices");
		for (int i = 0; i < 2; i++) {
			System.out.println(idx[i]);
		}
		
		Assert.assertArrayEquals(expected_output, outputD, 1e-6);
	}
}
