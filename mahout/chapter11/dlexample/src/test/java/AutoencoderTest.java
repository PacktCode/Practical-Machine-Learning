package test.java;

import java.io.Serializable;
import java.util.ArrayList;

import main.java.AutoencoderFctGrd;
import main.java.AutoencoderGradient3;
import main.java.AutoencoderParams;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.netlib.util.doubleW;

import scala.Tuple2;

public class AutoencoderTest implements Serializable{

	private transient JavaSparkContext sc;
	
	
	@Before
	public void setUp() throws Exception{
		sc = new JavaSparkContext("local", "AutoencoderTest");
	}
	
	@After
	public void tearDown() throws Exception{
		sc.close();
		sc = null;
	}
	
	@Ignore  @Test
	public void testGradient(){
		//this no longer valid - replace null with a correct configuration object
		ArrayList<Vector> trainExamples = new ArrayList<Vector>();
		trainExamples.add(new DenseVector(new double[]{1.0, 2.0, 3.0, 4.0}));
		trainExamples.add(new DenseVector(new double[]{2.0, 1.0, 3.0, 4.0}));
		trainExamples.add(new DenseVector(new double[]{4.0, 3.0, 2.0, 1.0}));
		trainExamples.add(new DenseVector(new double[]{1.0, 2.0, 1.0, 1.0}));
		
		JavaRDD<Vector> singleData = sc.parallelize(trainExamples);
//		JavaRDD<Tuple2<Vector, Vector>> trainData = singleData.map(new Function<Vector,Tuple2<Vector,Vector>>() {
//
//			@Override
//			public Tuple2<Vector, Vector> call(Vector arg0) throws Exception {
//				// TODO Auto-generated method stub
//				return new Tuple2(arg0,arg0);
//			}
//		});
		
		DenseMatrix W1 = new DenseMatrix(6, 4, new double[]{0.01,0.01,0.01,0.01,0.01,0.01,
															0.01,0.01,0.02,0.02,0.02,0.02,
															0.02,0.03,0.03,0.03,0.01,0.01,
															0.01,0.01,0.01,0.01,0.01,0.01});
		
		DenseMatrix W2 = new DenseMatrix(4,6, new double[]{ 0.01,0.01,0.01,0.02,
															0.01,0.01,0.01,0.01,
															0.01,0.01,0.01,0.01,
															0.01,0.01,0.01,0.01,
															0.02,0.02,0.02,0.02,
															0.02,0.02,0.02,0.02});
		
		DenseVector b1 = new DenseVector(new double[]{0.1,0.2,0.3,0.1,0.2,0.3});
		
		DenseVector b2 = new DenseVector(new double[]{0.1,0.2,0.3,0.1});
		
		Broadcast<AutoencoderParams> params = sc.broadcast(new AutoencoderParams(W1, W2, b1, b2));
		AutoencoderGradient3 autoencoderGradient = new AutoencoderGradient3(params, singleData,null);
		AutoencoderFctGrd grdFct = autoencoderGradient.getGradient();
		
		Assert.assertEquals(0.03,W1.apply(2, 2),1e-4);
		Assert.assertEquals(0.02,W1.apply(2, 1),1e-4);
		Assert.assertEquals(31.497532302255600,grdFct.getValue(),1e-9);
		Assert.assertEquals(6.012200834302241,grdFct.getW1().apply(0, 0),1e-6);
		Assert.assertEquals(7.544980734856909,grdFct.getW1().apply(2, 2),1e-6);	
		Assert.assertEquals(-0.201734482313380,grdFct.getW2().apply(0, 0),1e-6);	
		Assert.assertEquals(-0.248337911115040,grdFct.getW2().apply(2, 2),1e-6);
		Assert.assertEquals(-0.354249020371895,grdFct.getB2().apply(1),1e-6);	
		Assert.assertEquals(3.189070906280792,grdFct.getB1().apply(1),1e-6);	
	}
	
	
}
