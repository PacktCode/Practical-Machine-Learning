// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Vector;

import scala.util.Random;

public class AutoencoderLinAlgebra {
	
	//TO DO replace the calls to matrix multiply by DenseMatrix.multiply
	
	public static double sigmoid(double x){
//		if (x>25){
//			return (Math.exp(-Math.log(1+Math.exp(-x))));
//		}else if(x<-25){
//			return Math.exp(x - Math.log(1+Math.exp(x)));
//		}else{
//			return 1.0/(1+Math.exp(-x));
//		}
		return 1.0/(1+Math.exp(-x));
	}

	public static double[] MVM(DenseMatrix m, Vector v) throws Exception{
		int cols = v.size();
	    int rows = m.numRows();
	    if (cols!=m.numCols()){
	    	throw new Exception("MVM: Matrix and vector have inconsitent size");
	    };
	    double[] result = new double[rows];
	    for (int i=0;i<rows;i++){
	    	double partialSum = 0;
	    	for(int j=0;j<cols;j++){
	    		partialSum += m.apply(i, j)*v.apply(j);
	    		//System.out.println("M["+i+","+j+"]="+m.apply(i, j)+" * v["+j+"]="+v.apply(j));
	    	}
	    	result[i]= partialSum;
	    	//System.out.println("R["+i+"]="+result[i]);
	    }
	    
		return result;
	}
	
	public static double[] MAM(DenseMatrix m, double[] v) throws Exception{
		int cols = v.length;
	    int rows = m.numRows();
	    if (cols!=m.numCols()){
	    	throw new Exception("MVM: Matrix and vector have inconsitent size");
	    };
	    double[] result = new double[rows];
	    for (int i=0;i<rows;i++){
	    	double partialSum = 0;
	    	for(int j=0;j<cols;j++){
	    		partialSum += m.apply(i, j)*v[j];
	    	}
	    	result[i]= partialSum;
	    }
	    
		return result;
	}
	//to check
	public static double[] MTVM(DenseMatrix m, Vector v) throws Exception{
	
		int cols = v.size();
	    int rows = m.numCols();
	    if (cols!=m.numRows()){
	    	throw new Exception("MTVM: Matrix and vector have inconsitent size");
	    };
	    double[] result = new double[rows];
	    for (int i=0;i<rows;i++){
	    	double partialSum = 0;
	    	for(int j=0;j<cols;j++){
	    		partialSum += m.apply(j, i)*v.apply(j);
	    	}
	    	result[i]= partialSum;
	    }
	    
		return result;
	}
	
	public static double[] MTAM(DenseMatrix m, double[] v) throws Exception{
		
		int cols = v.length;
	    int rows = m.numCols();
	    if (cols!=m.numRows()){
	    	throw new Exception("MTVM: Matrix and vector have inconsitent size");
	    };
	    double[] result = new double[rows];
	    for (int i=0;i<rows;i++){
	    	double partialSum = 0;
	    	for(int j=0;j<cols;j++){
	    		partialSum += m.apply(j, i)*v[j];
	    	}
	    	result[i]= partialSum;
	    }
	    
		return result;
	}
	
	public static double[] VVA(Vector v1,Vector v2) throws Exception{
		int len = v1.size();
		if (v2.size()!=len){
			throw new Exception("VVA: Inconsistent sizes");
		};
		double[] result= new double[len];
		for(int i=0;i<len;i++){
			result[i] = v1.apply(i)+v2.apply(i);
		}
		return result;
	}

	public static double[] VAA(Vector v1,double[] v2) throws Exception{
		int len = v1.size();
		if (v2.length!=len){
			throw new Exception("VAA:Inconsistent sizes");
		};
		double[] result= new double[len];
		for(int i=0;i<len;i++){
			result[i] = v1.apply(i)+v2[i];
		}
		return result;
	}
	
	public static double[] AAA(double[] v1,double[] v2) throws Exception{
		int len = v1.length;
		if (v2.length!=len){
			throw new Exception("AAA:Inconsistent sizes:"+len+"<>"+v2.length);
		};
		double[] result= new double[len];
		for(int i=0;i<len;i++){
			result[i] = v1[i]+v2[i];
		}
		return result;
	}

	public static AutoencoderParams initialize(int num_input, int num_hidden) {
		Random r = new Random(System.currentTimeMillis());
		double dev = Math.sqrt(6) / Math.sqrt(num_hidden + num_input + 1);
		DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_hidden;i++){
			for (int j=0;j<num_input;j++){
				w1.update(i, j, r.nextDouble()*2*dev-dev);
			}
		}
		
		DenseMatrix w2 = new DenseMatrix(num_input,num_hidden , 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_input;i++){
			for (int j=0;j<num_hidden;j++){
				w2.update(i, j, r.nextDouble()*2*dev-dev);
			}
		};
		
		DenseVector b1 = new DenseVector(new double[num_hidden]);
		DenseVector b2 = new DenseVector(new double[num_input]);
		
		return new AutoencoderParams(w1, w2, b1, b2);
	}

	public static AutoencoderParams updateInitial(AutoencoderFctGrd result, double alpha) {
		int num_input = result.getW1().numCols();
		int num_hidden = result.getW1().numRows();
		
		DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_hidden;i++){
			for (int j=0;j<num_input;j++){
				w1.update(i, j, result.getW1().apply(i, j)*alpha);
			}
		}
		
		
		DenseMatrix w2 = new DenseMatrix(num_input, num_hidden, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_input;i++){
			for (int j=0;j<num_hidden;j++){
				w2.update(i, j, result.getW2().apply(i, j)*alpha);
			}
		}
		
		double[] b1A = new double[num_hidden];
		for (int i=0;i<num_hidden;i++){
			b1A[i] = result.getB1().apply(i)*alpha;
		}
		
		DenseVector b1 = new DenseVector(b1A);
		
		double[] b2A = new double[num_input];
		for (int i=0;i<num_input;i++){
			b2A[i] = result.getB2().apply(i)*alpha;
		}
		
		DenseVector b2 = new DenseVector(b2A);
		
		return new AutoencoderParams(w1, w2, b1, b2);
	}

	public static AutoencoderParams update(AutoencoderParams oldGrad,
			AutoencoderFctGrd result, double alpha, double momentum) {
		
		int num_input = result.getW1().numCols();
		int num_hidden = result.getW1().numRows();
		
		DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_hidden;i++){
			for (int j=0;j<num_input;j++){
				w1.update(i, j, oldGrad.getW1().apply(i, j)*momentum+ result.getW1().apply(i, j)*alpha);
			}
		}
		
		
		DenseMatrix w2 = new DenseMatrix(num_input, num_hidden, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_input;i++){
			for (int j=0;j<num_hidden;j++){
				w2.update(i, j, oldGrad.getW2().apply(i, j)*momentum + result.getW2().apply(i, j)*alpha);
			}
		}
		
		double[] b1A = new double[num_hidden];
		for (int i=0;i<num_hidden;i++){
			b1A[i] = oldGrad.getB1().apply(i)*momentum + result.getB1().apply(i)*alpha;
		}
		
		DenseVector b1 = new DenseVector(b1A);
		
		double[] b2A = new double[num_input];
		for (int i=0;i<num_input;i++){
			b2A[i] = oldGrad.getB2().apply(i)*momentum +result.getB2().apply(i)*alpha;
		}
		
		DenseVector b2 = new DenseVector(b2A);
		
		return new AutoencoderParams(w1, w2, b1, b2);
	}

	public static AutoencoderParams updateParams(AutoencoderParams params,
			AutoencoderParams oldGrad) {
		
		int num_input = params.getW1().numCols();
		int num_hidden = params.getW1().numRows();
		
		DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_hidden;i++){
			for (int j=0;j<num_input;j++){
				w1.update(i, j, params.getW1().apply(i, j) -  oldGrad.getW1().apply(i, j));
			}
		}
		
		
		DenseMatrix w2 = new DenseMatrix(num_input, num_hidden, 
				new double[num_hidden * num_input]);
		for (int i=0;i<num_input;i++){
			for (int j=0;j<num_hidden;j++){
				w2.update(i, j, params.getW2().apply(i, j) - oldGrad.getW2().apply(i, j));
			}
		}
		
		double[] b1A = new double[num_hidden];
		for (int i=0;i<num_hidden;i++){
			b1A[i] = params.getB1().apply(i) -  oldGrad.getB1().apply(i);
		}
		
		DenseVector b1 = new DenseVector(b1A);
		
		double[] b2A = new double[num_input];
		for (int i=0;i<num_input;i++){
			b2A[i] = params.getB2().apply(i) - oldGrad.getB2().apply(i);
		}
		
		DenseVector b2 = new DenseVector(b2A);
		
		return new AutoencoderParams(w1, w2, b1, b2);
	}
	
	public static Vector[] getFilters(AutoencoderParams params){
		int num_input = params.getW1().numCols();
		int num_hidden = params.getW1().numRows();
		Vector[] result = new Vector[num_hidden];
		
		for(int i=0;i<num_hidden;i++){
			double[] row = new double[num_input];
			for(int j=0;j<num_input;j++){
				row[j] = params.getW1().apply(i, j);
			}
			result[i] = new DenseVector(row);
		}
		return result;
	}

	public static double DP(DenseMatrix w1, DenseMatrix w2) throws Exception {
		int cols = w1.numCols();
	    int rows = w1.numRows();
	    if (w2.numCols()!=cols || w2.numRows()!=rows){
			throw new Exception("VAA:Inconsistent sizes");
		};
		double result = 0;
		for(int i=0;i<rows;i++){
			for (int j=0;j<cols;j++){
				result += w1.apply(i,j)*w2.apply(i, j);
			}
		}
		return result;
	}

	public static DenseMatrix MMAM(DenseMatrix w1, DenseMatrix w2, double alpha) throws Exception {
		int cols = w1.numCols();
	    int rows = w1.numRows();
	    if (w2.numCols()!=cols || w2.numRows()!=rows){
			throw new Exception("VAA:Inconsistent sizes");
		};
		DenseMatrix result = new DenseMatrix(rows, cols, 
				new double[rows*cols]);
		for (int i=0;i<rows;i++){
			for (int j=0;j<cols;j++){
				result.update(i, j, w1.apply(i, j) + alpha* w2.apply(i, j));
			}
		}
		return result;
	}

	public static DenseVector VVAM(DenseVector b1, DenseVector b2, double alpha) throws Exception {
	    int rows = b1.size();
	    if ( b2.size() !=rows){
			throw new Exception("VAA:Inconsistent sizes");
		};
		double[] result = new double[rows];
		for (int i=0;i<rows;i++){			
				result[i] = b1.apply(i) + alpha* b2.apply(i);
		
		}
		return new DenseVector(result);
	}
}
