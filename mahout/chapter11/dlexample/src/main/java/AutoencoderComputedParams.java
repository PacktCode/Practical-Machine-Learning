// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import java.io.Serializable;

public class AutoencoderComputedParams implements Serializable {
	
	private long numSamples;
	private double[] sparsityArray;
	
	public AutoencoderComputedParams(long numSamples, double[] sparsityArray) {
		super();
		this.numSamples = numSamples;
		this.sparsityArray = sparsityArray;
	}

	public long getNumSamples() {
		return numSamples;
	}

	public void setNumSamples(long numSamples) {
		this.numSamples = numSamples;
	}

	public double[] getSparsityArray() {
		return sparsityArray;
	}

	public void setSparsityArray(double[] sparsityArray) {
		this.sparsityArray = sparsityArray;
	}


	
	
}
