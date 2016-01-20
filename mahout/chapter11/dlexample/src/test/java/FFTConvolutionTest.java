package test.java;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import main.java.ConvMultiplyExtractor;
import main.java.DeepModelSettings.ConfigFeatureExtractor.NonLinearity;
import main.java.Extractor;
import main.java.FFTConvolution;
import main.java.FFTConvolutionExtractor;
import main.java.MatrixOps;
import main.java.PreProcessZCA;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DeepModelSettings.ConfigPreprocess;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class FFTConvolutionTest implements Serializable {
	
	private static final long serialVersionUID = 2780960378816038954L;
	private transient JavaSparkContext sc;
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "FFTConvolutionTest");
	}

	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}

	public boolean deepApproximateEquals(double[][][] x, double[][][] y, double epsilon) {
		if(x.length != y.length) fail("1D lengths differ");
		for(int i = 0; i < x.length; ++i) {
			if(x[i].length != y[i].length) fail("2D lengths differ at " + i);
			for(int j = 0; j < x[i].length; ++j) {
				if(x[i][j].length != y[i][j].length) fail("3D lengths differ at " + i + " , " + j);
				for(int k = 0; k < x[i][j].length; ++k) {
					if(Math.abs(x[i][j][k] - y[i][j][k]) > epsilon ) {
						fail("Values at " + i + "-" + j + "-" + k + " differ: " +  x[i][j][k] + " vs. " + y[i][j][k]);
						return false;
					}
				}
			}
		}
		return true;
	}
	
	@Test @Ignore
	public void simpleTest1DFFT() {
		double[][] input =   {{1,2,3,0}};
		double[][][] output = FFTConvolution.fftReal1Dor2D(input);
		double[][][] expected = {{{6,0},{-2,-2},{2,0},{-2,2}}};
		//System.out.println(Arrays.deepToString(output));
		assertTrue(deepApproximateEquals(output, expected, 1e-2));
	}
	
	@Test @Ignore
	public void simpleTest2DFFT() {
		double[][] input =   {{1,2,3,0}, {4,5,6,0}, {7,8,9,0},{0,0,0,0}};
		double[][][] output = FFTConvolution.fftReal1Dor2D(input);
		//System.out.println(Arrays.deepToString(output));
		double[][][] expected = {{{ 45,  0}, { -6,-15}, { 15,  0}, { -6, 15}},
				 {{-18,-15}, { -5,  8}, { -6, -5}, {  5, -4}},
				 {{ 15,  0}, { -2, -5}, {  5,  0}, { -2,  5}},
				 {{-18, 15}, {  5,  4}, { -6,  5}, { -5, -8}}};

		assertTrue(deepApproximateEquals(output, expected, 1e-2));
	}
	
	@Test
	public void simpleTest1DConvolution() {
		int inputRows = 1;
		int inputCols = 4;
		int featureRows = 1;
		int featureCols = 2;
		double[] input =   {1,2,3,0};
	    
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(featureCols).setFeatureDim2(featureRows).
						                  setInputDim1(inputCols).setInputDim2(inputRows)).
			    setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).build();
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf); // no pre-processing yet

		Vector data = new DenseVector(input);
		
		double[] A = {1,2};
		
		Vector[] features = {new DenseVector(A)};
		extractor.setFeatures(features);
		Vector output;
		try {
			output = extractor.call(data);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		//System.out.println(Arrays.toString(output.toArray()));
		double[] expected_outputs = {5,8,3};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	
	@Test
	public void simpleTest2DConvolution() {
		int inputRows = 4;
		int inputCols = 4;
		int featureRows = 2;
		int featureCols = 2;
		double[] input =   {1,2,3,0,
							4,5,6,0,
							7,8,9,0,
							0,0,0,0};
	    
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(featureCols).setFeatureDim2(featureRows).
						                  setInputDim1(inputCols).setInputDim2(inputRows)).
			    setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).build();
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf); // no pre-processing yet

		Vector data = new DenseVector(input);
		
		double[] A = {1,2,
					  0,0};
		
		Vector[] features = {new DenseVector(A)};
		extractor.setFeatures(features);
		Vector output;
		try {
			output = extractor.call(data);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		//System.out.println(Arrays.toString(output.toArray()));
		double[] expected_outputs = {5,8,3,
									 14,17,6,
									 23,26,9};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	
	
	/*
	 * The next four tests test the whole extractor with different parts of 
	 * preprocessing. The last is a total integration-style test.
	 * The reference values are calculated using MATLAB, something like:
	 * 
	 * 
	    zcaR = [0.9575    0.9572    0.4218    0.6557;
                0.9649    0.4854    0.9157    0.0357;
                0.1576    0.8003    0.7922    0.8491;
                0.9706    0.1419    0.9595    0.9340];

		meanR = [0.6324   0.0975    0.2785    0.5469];
		
		meanI = zeros(1,4);
		zcaI = diag(ones(4,1));
		
		x = [0.5600    0.3400    0.3200    0.1400;
		     0.5400    0.6300    1.2000    0.7800;
		     1.2300    0.3400    0.6700    0.8500;
		     0.5700    0.8500    0.2900    0.9400];
		
		f1 = [0.1000    0.2000;
		      0.4000    1.4000];
		
		f2 = [0.5000    0.2000;
		      0.1000    0.5000];
		
		r = zeros(3);
		for j = 1:3
		    for i=1:3
		        r(j,i) = conv(zcaR*(reshape(x(j:j+1,i:i+1)',1,4) - meanR)', flip(reshape(f2',1,4)), 'valid');
		    end
		end
	 * 
	 */
	
	@Test
	public void identityPreprocessingFFTConvTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] x = {0.56,   0.34,   0.32,   0.14,
				   	  0.54,   0.63,   1.20,   0.78,
				      1.23,   0.34,   0.67,   0.85,
				      0.57,   0.85,   0.29,   0.94};
		
		double[] zca = {1.0,0.0,0.0,0.0,
						0.0,1.0,0.0,0.0,
						0.0,0.0,1.0,0.0,
						0.0,0.0,0.0,1.0};
		double[] m = {0.0, 0.0, 0.0, 0.0};

		double[] expected_output = {1.2220, 2.0300, 1.6320, 
	    							1.1480, 1.3770, 1.7340,
	    							1.6090, 0.9140, 1.6690};
	    
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca);
		
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf);
		extractor.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
 		Vector[] vf = {Vectors.dense(f1)};
		extractor.setFeatures(vf);
		Vector out;
		try {
			out = extractor.call(Vectors.dense(x));
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		
		//System.out.println(Arrays.toString(out.toArray()));
		
		Assert.assertArrayEquals(expected_output, out.toArray(), 1e-6);	
	}
	
	@Test
	public void identityZcaPreprocessingFFTConvTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] x = {0.56,   0.34,   0.32,   0.14,
				   	  0.54,   0.63,   1.20,   0.78,
				      1.23,   0.34,   0.67,   0.85,
				      0.57,   0.85,   0.29,   0.94};
		
		double[] zca = {1.0,0.0,0.0,0.0,
						0.0,1.0,0.0,0.0,
						0.0,0.0,1.0,0.0,
						0.0,0.0,0.0,1.0};
		double[] m = {0.6324, 0.0975, 0.2785, 0.5469};

		double[] expected_output = {0.2622, 1.0702, 0.6722, 
	    							0.1882, 0.4172, 0.7742,
	    							0.6492,-0.0458, 0.7092};
	    
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca).transpose(); // just for principle - col vs row major
		
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf);
		extractor.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
 		Vector[] vf = {Vectors.dense(f1)};
		extractor.setFeatures(vf);
		Vector out;
		try {
			out = extractor.call(Vectors.dense(x));
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		
		//System.out.println(Arrays.toString(out.toArray()));
		
		Assert.assertArrayEquals(expected_output, out.toArray(), 1e-2);
	}
	
	@Test
	public void zeroMeanPreprocessingFFTConvTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] x = {0.56,   0.34,   0.32,   0.14,
				   	  0.54,   0.63,   1.20,   0.78,
				      1.23,   0.34,   0.67,   0.85,
				      0.57,   0.85,   0.29,   0.94};
		
		double[] zca = {0.9575,   0.9572,   0.4218,   0.6557,
			            0.9649,   0.4854,   0.9157,   0.0357,
			            0.1576,   0.8003,   0.7922,   0.8491,
			            0.9706,   0.1419,   0.9595,   0.9340};
		
		double[] m = {0.0, 0.0, 0.0, 0.0};

		double[] expected_output = {3.3016,   4.0611,   4.2512,
			    					4.2759,   3.7250,   5.3330,
			    					4.8827,   3.1598,   3.9145};
	    
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca).transpose(); // Matrix creation is column-major, ours is row-major
		
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf);
		extractor.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
 		Vector[] vf = {Vectors.dense(f1)};
		extractor.setFeatures(vf);
		Vector out;
		try {
			out = extractor.call(Vectors.dense(x));
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		
		//System.out.println(Arrays.toString(out.toArray()));
		
		Assert.assertArrayEquals(expected_output, out.toArray(), 1e-2);	
	}
	
	@Test
	public void preprocessingFFTConvTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		// simple example
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] f2 = {0.5, 0.2, 0.1, 0.5};
		double[] x = {0.560000000000000,   0.340000000000000,   0.320000000000000,   0.140000000000000,
				   0.540000000000000,   0.630000000000000,   1.200000000000000,   0.780000000000000,
				   1.230000000000000,   0.340000000000000,   0.670000000000000,   0.850000000000000,
				   0.570000000000000,   0.850000000000000,   0.290000000000000,   0.940000000000000};
		
		double[] zca = {0.9575,   0.9572,   0.4218,   0.6557,
	            0.9649,   0.4854,   0.9157,   0.0357,
	            0.1576,   0.8003,   0.7922,   0.8491,
	            0.9706,   0.1419,   0.9595,   0.9340};

		double[] m = {0.6324, 0.0975, 0.2785, 0.5469};

		double[] expected_output = {0.6848, 1.4443, 1.6343,
									1.6591, 1.1081, 2.7161,
									2.2658, 0.5430, 1.2976,
									
									0.4142,    0.7330,    0.7495,
								    1.0019,    0.9661,    1.8036,
								    1.4237,    0.3899,    0.9505};
	    
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca).transpose();
		
		// create a PreProcessZCA object with the input mean and ZCA variables
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA, conf);
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(1);
		matX.add(Vectors.dense(x));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// create the array of feature vectors
 		Vector[] vf = {Vectors.dense(f1), Vectors.dense(f2)};
		
		// create a MultiplyExtractor object
 		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf);
		extractor.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
		extractor.setFeatures(vf);
	
		// call the feature extraction process
		matRDD = matRDD.map(extractor);
		
		Vector[] outputD = matRDD.collect().toArray(new Vector[1]);
		DenseMatrix outputM = MatrixOps.convertVectors2Mat(outputD);
		//System.out.println(Arrays.toString(outputM.toArray()));
		
		Assert.assertArrayEquals(expected_output, outputM.toArray(), 1e-2);		
	}
	
	@Test
	public void convMultiplyTest() {
		
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4).setNonLinearity(NonLinearity.ABS)).
		setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).
		setConfigPreprocess(ConfigPreprocess.newBuilder().setEps1(0.1).setEps2(0.1)).build();
		
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] f2 = {0.5, 0.2, 0.1, 0.5};
		double[] x = {0.560000000000000,   0.340000000000000,   0.320000000000000,   0.140000000000000,
				   0.540000000000000,   0.630000000000000,   1.200000000000000,   0.780000000000000,
				   1.230000000000000,   0.340000000000000,   0.670000000000000,   0.850000000000000,
				   0.570000000000000,   0.850000000000000,   0.290000000000000,   0.940000000000000};
		
		double[] zca = {2.262239321973017,   0.366216542149525,   0.009718022083399,  -0.269118962890450,
		   		0.366216542149524,   2.670819891727144,   0.051028161893375,  -0.413249779370649,
		   		0.009718022083399,   0.051028161893375,   1.918559331530568,  -0.201107593595758,
		   		-0.269118962890450,  -0.413249779370648,  -0.201107593595758,   2.216674270547634};
		
		double[] m = {0.725000000000000,   0.540000000000000,   0.620000000000000,   0.677500000000000};
		
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
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf);
		extractor.setPreProcessZCA(preProcess.getZCA(), preProcess.getMean());
		extractor.setFeatures(vf);
	
		// call the feature extraction process
		matRDD = matRDD.map(extractor);
		
		Vector[] outputD = matRDD.collect().toArray(new Vector[1]);
		DenseMatrix outputM = MatrixOps.convertVectors2Mat(outputD);

		System.out.println(Arrays.toString(expected_output));
		System.out.println(Arrays.toString(outputM.toArray()));
		Assert.assertArrayEquals(expected_output, outputM.toArray(), 1e-3);
		
	}
	
}
