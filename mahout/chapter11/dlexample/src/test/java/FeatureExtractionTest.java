package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import main.java.ConvMultiplyExtractor;
import main.java.DeepModelSettings.ConfigPreprocess;
import main.java.MatrixOps;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DeepModelSettings.ConfigFeatureExtractor.NonLinearity;
import main.java.MultiplyExtractor;
import main.java.PreProcessZCA;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class FeatureExtractionTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5361837911584977475L;
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
	public void multiplyTest() {
		//ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().setConfigFeatureExtractor()
		
		// simple example
		double[] x = {0.1, 0.2, 0.4, 1.4, 2.3};
		double[] f1 = {0.56, 0.34, 0.32, 0.14, 0.25};
		double[] f2 = {0.54, 0.63, 1.2, 0.78, 1.23};
		double[] f3 = {1.23, 0.34, 0.67, 0.85, 0.43};
		double[] expected_output = {1.0230, 4.5810, 2.6380};
		
		// create Vectors from double arrays
		Vector vx = Vectors.dense(x);
		DenseVector dvx = (DenseVector) vx;
		
 		Vector[] vf = new Vector[3];
		vf[0] = Vectors.dense(f1);
		vf[1] = Vectors.dense(f2);
		vf[2] = Vectors.dense(f3);
	
		// run the feature extraction code
		DenseMatrix D = MatrixOps.convertVectors2Mat(vf);
		DenseVector dvxOut = new DenseVector(new double[D.numRows()]);
		BLAS.gemv(1.0, D, dvx, 0.0, dvxOut);
		
		Assert.assertArrayEquals(expected_output, dvxOut.toArray(), 1e-6);
		
	}
	
	
	@Test
	public void multiplyPreTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4).setNonLinearity(ConfigFeatureExtractor.NonLinearity.SOFT).setSoftThreshold(0.1)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		// simple example
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] f2 = {0.5, 0.2, 0.1, 0.5};
		double[] x1 = {0.35, 0.65, 0.28, 0.12}; 
		double[] x2 = {0.86, 0.96, 0.34, 0.57};
		//double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		
		//double[] zca = {1.654633794518243,   0.541992148747697,   0.519961336130961,   0.445690380771477,
		//				0.541992148747697,   1.919200272146810,   0.522623043139470,   0.178462196134402,
		//				0.519961336130961,   0.522623043139470,   1.601056255503961,   0.518637025393985,
		//				0.445690380771477,   0.178462196134402,   0.518637025393985,   2.019488057868512};	
		double[] zca = {2.262239321973017,   0.366216542149525,   0.009718022083399,  -0.269118962890450,
				   		0.366216542149524,   2.670819891727144,   0.051028161893375,  -0.413249779370649,
				   		0.009718022083399,   0.051028161893375,   1.918559331530568,  -0.201107593595758,
				   		-0.269118962890450,  -0.413249779370648,  -0.201107593595758,   2.216674270547634};
		
		//double[] m = {0.190168010867027,  -0.204704063143829,  -0.042614218658781,   0.057150270935583};
		double[] m = {0.725000000000000,   0.540000000000000,   0.620000000000000,   0.677500000000000};
		
		//double[] expected_output = {-0.145702567855953,  1.591192624037388,  -0.146329897991626,  -1.299160158189805,
		//							 0.302514463615425,	 1.316131165465278,  -0.825195898488151,  -0.793449730592545};
		//double[] expected_output = {-1.573687912640495,  -1.147430302770226,
		//							-0.418825828014564,  -0.064760990244320};
		//double[] expected_output = {-1.764262994854122,  -0.456956908917123,
		//		  					-0.866564862855998,   0.237752718826297};
		double[] expected_output = {0, 0, 0, 0.137752718826297};
		
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca);
		
		// create a PreProcessZCA object with the input mean and ZCA variables
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(4);
		matX.add(Vectors.dense(x1));
		matX.add(Vectors.dense(x2));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// create the array of feature vectors
 		Vector[] vf = new Vector[2];
		vf[0] = Vectors.dense(f1);
		vf[1] = Vectors.dense(f2);
		
		// create a MultiplyExtractor object
		MultiplyExtractor multi = new MultiplyExtractor(conf);
		multi.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
		multi.setFeatures(vf);
		
		// call the feature extraction process
		matRDD = matRDD.map(multi);
		
		Vector[] outputD = matRDD.collect().toArray(new Vector[2]);
		DenseMatrix outputM = MatrixOps.convertVectors2Mat(outputD);
		
		Assert.assertArrayEquals(expected_output, outputM.toArray(), 1e-6);
		
	}
	
	
	@Test
	public void convMultiplyTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4).setNonLinearity(NonLinearity.ABS)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		// simple example
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] f2 = {0.5, 0.2, 0.1, 0.5};
		double[] x = {0.560000000000000,   0.340000000000000,   0.320000000000000,   0.140000000000000,
				   0.540000000000000,   0.630000000000000,   1.200000000000000,   0.780000000000000,
				   1.230000000000000,   0.340000000000000,   0.670000000000000,   0.850000000000000,
				   0.570000000000000,   0.850000000000000,   0.290000000000000,   0.940000000000000};
		
		//double[] zca = {1.654633794518243,   0.541992148747697,   0.519961336130961,   0.445690380771477,
		//		0.541992148747697,   1.919200272146810,   0.522623043139470,   0.178462196134402,
		//		0.519961336130961,   0.522623043139470,   1.601056255503961,   0.518637025393985,
		//		0.445690380771477,   0.178462196134402,   0.518637025393985,   2.019488057868512};
		double[] zca = {2.262239321973017,   0.366216542149525,   0.009718022083399,  -0.269118962890450,
		   		0.366216542149524,   2.670819891727144,   0.051028161893375,  -0.413249779370649,
		   		0.009718022083399,   0.051028161893375,   1.918559331530568,  -0.201107593595758,
		   		-0.269118962890450,  -0.413249779370648,  -0.201107593595758,   2.216674270547634};
		
		//double[] m = {0.190168010867027,  -0.204704063143829,  -0.042614218658781,   0.057150270935583};
		double[] m = {0.725000000000000,   0.540000000000000,   0.620000000000000,   0.677500000000000};
		
		//double[] expected_output = {1.1310, 1.4080, 2.1030, 0.9120, 1.3250, 0.9790, 1.0340, 2.1890, 1.3180};
	    //double[] expected_output = {0.662032762310817,   2.228330709948044, 1.059039882521740, -1.054450834512288, -0.845244929924908, -0.436701601060417, 0.296864933161854, -1.091676319630362, 0.579432074562157, 
	    //		0.115146987448164, 0.248682789409449, -0.176343125716004, -0.607612501991995, -0.212833806576546, 0.208543928476519, 0.405416463911857, -0.628706789196375, 0.194667022197182,};
	    double[] expected_output = {0.168564094601626,   1.552671452042367,   0.611579585063065,  0.665331628736182,  0.145702887838090,   0.495568102348595,   0.438095915047013, 0.984689478972046,   0.608683133315226,
	    		0.329710605029764,  0.069543507208860,  0.488992094416595,  0.380760393962722,   0.202356138166303,   0.787967065953481,   0.582523569274661, 0.658255668206114,   0.294530822631855};
	    
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca);
		
		// create a PreProcessZCA object with the input mean and ZCA variables
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(1);
		matX.add(Vectors.dense(x));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// create the array of feature vectors
 		Vector[] vf = new Vector[2];
		vf[0] = Vectors.dense(f1);
		vf[1] = Vectors.dense(f2);
		
		// create a MultiplyExtractor object
		ConvMultiplyExtractor multi = new ConvMultiplyExtractor(conf);
		multi.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
		multi.setFeatures(vf);
	
		// call the feature extraction process
		matRDD = matRDD.map(multi);
		
		Vector[] outputD = matRDD.collect().toArray(new Vector[1]);
		DenseMatrix outputM = MatrixOps.convertVectors2Mat(outputD);
		
//		// run the feature extraction code
//		DenseMatrix D = MatrixOps.convertVectors2Mat(vf);
//		int[] dims = {4,4};
//		int[] rfSize = {2,2};
//		DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) vx, dims);	
//		DenseMatrix patches = MatrixOps.im2colT(M, rfSize);
//		
//		// allocate memory for the output vector
//		DenseMatrix out = new DenseMatrix(patches.numRows(),D.numRows(),new double[patches.numRows()*D.numRows()]);	
//		// multiply the matrix of the learned features with the preprocessed data point
//		BLAS.gemm(1.0, patches, D.transpose(), 0.0, out);
//		//DenseVector outVec = MatrixOps.reshapeMat2Vec(out);
		
		Assert.assertArrayEquals(expected_output, outputM.toArray(), 1e-6);
		
	}
	
}
