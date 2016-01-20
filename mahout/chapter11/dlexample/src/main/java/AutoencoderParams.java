// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import java.io.Serializable;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;

public class AutoencoderParams implements Serializable {
	
	private DenseMatrix w1;
	private DenseMatrix w2;
	private DenseVector b1;
	private DenseVector b2;
	
	public AutoencoderParams(DenseMatrix w1, DenseMatrix w2, DenseVector b1, DenseVector b2) {
		this.w1 = w1;
		this.w2 = w2;
		this.b1 = b1;
		this.b2 = b2;
	}

	public DenseMatrix getW1() {
		return w1;
	}

	public void setW1(DenseMatrix w1) {
		this.w1 = w1;
	}

	public DenseMatrix getW2() {
		return w2;
	}

	public void setW2(DenseMatrix w2) {
		this.w2 = w2;
	}

	public DenseVector getB1() {
		return b1;
	}

	public void setB1(DenseVector b1) {
		this.b1 = b1;
	}

	public DenseVector getB2() {
		return b2;
	}

	public void setB2(DenseVector b2) {
		this.b2 = b2;
	}
	
	

}
