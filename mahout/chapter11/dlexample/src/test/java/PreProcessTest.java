package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import main.java.ContrastNormalization;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.MatrixOps;
import main.java.PreProcessZCA;
import main.java.PreProcessor;
import main.java.SubtractMean;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigPreprocess;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class PreProcessTest implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7766242513300032083L;
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
	public void preprocessTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		// simple test example for pre-processing
		double[] x1 = {0.56, 0.34, 0.32, 0.14};
		double[] x2 = {0.54, 0.63, 1.2, 0.78};
		double[] x3 = {1.23, 0.34, 0.67, 0.85};
		double[] x4 = {0.57, 0.85, 0.29, 0.94};
		double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		//double[] expected_output = {-1.114153311705324, -0.705279543151641, -1.170587714869567, -1.330205911918481};
		
		
		//double[] expected_output = {1.654633794518243,   0.541992148747697,   0.519961336130961,   0.445690380771477,
		//							0.541992148747697,   1.919200272146810,   0.522623043139470,   0.178462196134402,
		//							0.519961336130961,   0.522623043139470,   1.601056255503961,   0.518637025393985,
		//							0.445690380771477,   0.178462196134402,   0.518637025393985,   2.019488057868512};
		
		//double[] expected_output = {0.527648586339945, 	-0.865959542426838, 0.826537951096369, 	-0.488226995009476, 
		//							0.504907680845764,  -0.214080196045214, -0.973763595467167, 0.682936110666617, 
		//							-0.012634538424828, 1.081367727655617,  -0.183203633132266, -0.885529556098523, 
		//							-1.019921728760881, -0.001327989183566, 0.330429277503064,  0.690820440441383};
		
		double[] expected_output = {  -0.304776760626856,  -0.407503026659451,   1.023250429172036,  -0.310970641885730,
				  -0.387776399956389,   0.159861961470447,  -0.417958803406687,   0.645873241892629,
				  -0.479281573923886,   1.094945584429139,   0.055938875454702,  -0.671602885959955,
				  -1.004075557589572,   0.123161236436968,   0.319065811604132,   0.561848509548473};
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(4);
		matX.add(Vectors.dense(x1));
		matX.add(Vectors.dense(x2));
		matX.add(Vectors.dense(x3));
		matX.add(Vectors.dense(x4));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// run pre-processing
		PreProcessZCA preProcess = new PreProcessZCA(conf);
		matRDD = preProcess.preprocessData(matRDD);
		
		Vector[] outputD = matRDD.collect().toArray(new Vector[4]);
		DenseMatrix outputM = MatrixOps.convertVectors2Mat(outputD);
		
		Assert.assertArrayEquals(expected_output, outputM.toArray(), 1e-6);
		
	}
	

}
