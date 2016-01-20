// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import main.java.DeepModelSettings.ConfigAutoencoders;
import main.java.DeepModelSettings.ConfigBaseLayer;

public class AutoencoderConfig {

	private double rho = 0.05;
	private double lambda = 0.001;
	private double beta = 6;

	private double alpha = 0.0005;
	private double momentum = 0.5;
	private double initMomentum = 0.5;
	private double increaseMomentum = 0.9;

	private int numEpochs  = 2;
	private int numBatches = 2;

	private int num_input  = 32*32;
	private int num_hidden = 1000;
	
	private int lineSearchStrategy = 0;
	
	private double[] alphaSteps = new double[]{0.05,2.0,10.0}; //initial value, decrease by, maximum number of iterations
	
	public AutoencoderConfig(ConfigBaseLayer config){
		ConfigAutoencoders conf = config.getConfigAutoencoders();
		if (conf.hasNumberOfUnits()){
			this.num_hidden = conf.getNumberOfUnits();
		}
		if (conf.hasRho()){
			this.rho = conf.getRho();
		}
		
		if (conf.hasLambda()){
			this.lambda = conf.getLambda();
		}
		if (conf.hasBeta()){
			this.beta = conf.getBeta();
		}
		if (conf.hasNumEpochs()){
			this.numEpochs = conf.getNumEpochs();
		}
		if (conf.hasNumBatches()){
			this.numBatches = conf.getNumBatches();
		}
		if (conf.hasLineSearchStrategy()){
			this.lineSearchStrategy = conf.getLineSearchStrategy();
		}
		if (conf.hasAlpha()){
			this.alpha = conf.getAlpha();
		}
		if (conf.hasMomentum()){
			this.momentum = conf.getMomentum();
		}
		if (conf.hasInitMomentum()){
			this.initMomentum = conf.getInitMomentum();
		}
		if (conf.hasIncreaseMomentum()){
			this.increaseMomentum = conf.getIncreaseMomentum();
		}
		if (conf.hasNumInput()){
			this.num_input = conf.getNumInput();
		}
		
		if (conf.hasAlphaInit()){
			this.alphaSteps[0] = conf.getAlphaInit();
		}
		if (conf.hasAlphaStep()){
			this.alphaSteps[1] = conf.getAlphaStep();
		}
		if (conf.hasAlphaMaxSteps()){
			this.alphaSteps[2] = (double) conf.getAlphaMaxSteps();
		}
	}

	public double getRho() {
		return rho;
	}

	public void setRho(double rho) {
		this.rho = rho;
	}

	public double getLambda() {
		return lambda;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public double getInitMomentum() {
		return initMomentum;
	}

	public void setInitMomentum(double initMomentum) {
		this.initMomentum = initMomentum;
	}

	public double getIncreaseMomentum() {
		return increaseMomentum;
	}

	public void setIncreaseMomentum(double increaseMomentum) {
		this.increaseMomentum = increaseMomentum;
	}

	public int getNumEpochs() {
		return numEpochs;
	}

	public void setNumEpochs(int numEpochs) {
		this.numEpochs = numEpochs;
	}

	public int getNumBatches() {
		return numBatches;
	}

	public void setNumBatches(int numBatches) {
		this.numBatches = numBatches;
	}

	public int getNum_input() {
		return num_input;
	}

	public void setNum_input(int num_input) {
		this.num_input = num_input;
	}

	public int getNum_hidden() {
		return num_hidden;
	}

	public void setNum_hidden(int num_hidden) {
		this.num_hidden = num_hidden;
	}

	public int getLineSearchStrategy() {
		return lineSearchStrategy;
	}

	public void setLineSearchStrategy(int lineSearchStrategy) {
		this.lineSearchStrategy = lineSearchStrategy;
	}

	public double[] getAlphaSteps() {
		return alphaSteps;
	}

	public void setAlphaSteps(double[] alphaSteps) {
		this.alphaSteps = alphaSteps;
	}
	
	
}
