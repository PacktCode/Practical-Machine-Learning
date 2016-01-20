// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;

public class Autoencoder {

	
	
	private JavaRDD<Vector> trainData;
	private JavaRDD<Vector> testData;
	private JavaSparkContext sc;
	private long train_size;
	private AutoencoderConfig conf;
	private AutoencoderParams params;
	private AutoencoderParams oldGrad;
	
	//Loaded from conf
	private double rho = 0.05;
	private double lambda = 0.001;
	private double beta = 6;

	private double alpha = 0.0005;
	private double momentum = 0.5;
	private double initMomentum = 0.5;
	private double increaseMomentum = 0.9;

	private int numEpochs = 2;
	private int numBatches = 2;

	private int num_input = 32*32;
	private int num_hidden = 1000;

	private double alpha_init = 0.05;
	private double alpha_decrease = 2.0;
	private int    alpha_max_steps = 10;

	
	
	
	public Autoencoder(AutoencoderConfig conf){
		this.conf = conf;
		
		this.rho = conf.getRho();
		this.lambda = conf.getLambda();
		this.beta = conf.getBeta();
		
		this.alpha = conf.getBeta();
		this.momentum = conf.getMomentum();
		this.initMomentum = conf.getMomentum();
		this.increaseMomentum = conf.getIncreaseMomentum();
		
		this.numEpochs = conf.getNumEpochs();
		this.numBatches = conf.getNumBatches();
		
		this.num_hidden = conf.getNum_hidden();
		
		double[] lineConf = conf.getAlphaSteps();
		this.alpha_init = lineConf[0];
		this.alpha_decrease = lineConf[1];
		this.alpha_max_steps = (int) Math.round(lineConf[2]);
	}
	
	public Vector[] train(JavaRDD<Vector> data) throws Exception{

		sc = JavaSparkContext.fromSparkContext(data.context());
		num_input = data.take(1).iterator().next().size();
		conf.setNum_input(num_input);
		
		//Split to train and test data
		split(data);
		train_size = trainData.count();
		
		params = initializeWeights();
		for (int i=0;i<numEpochs;i++){
			//suffle randomly data
			
			momentum = initMomentum;
			Broadcast<AutoencoderParams> brParams = null;
			oldGrad = null;
			for (int j=0;j<numBatches;j++){
				if(j==20){
					momentum = increaseMomentum;
				}
				 brParams = sc.broadcast(params);

				AutoencoderGradient3 gradient = new AutoencoderGradient3(brParams,trainData.sample(false, 1.0/ (double) numBatches),conf);
				AutoencoderFctGrd    result = gradient.getGradient();
				
				AutoencoderLineSearch linesearch = new AutoencoderLineSearch( brParams, sc.broadcast(result), 
						trainData.sample(false, 1.0/100.0/numBatches), conf);
				try {
					linesearch.precompute();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				boolean notFound = true;
				int steps = 0;
				alpha = alpha_init;
				double oldFctValue = linesearch.computeTestError(0.0);
				while(notFound && steps++<=alpha_max_steps){
					double curFctValue = linesearch.computeTestError(alpha);
					System.out.println("Line search:"+oldFctValue+" <> "+curFctValue);
					if (curFctValue<oldFctValue) notFound = true;
					alpha /= alpha_decrease;
				}

				if (oldGrad == null){
					oldGrad = AutoencoderLinAlgebra.updateInitial(result,alpha);
				}else{
					oldGrad = AutoencoderLinAlgebra.update(oldGrad,result,alpha,0.0);
				}

				params = AutoencoderLinAlgebra.updateParams(params,oldGrad);

				System.out.println("Epoch "+i+", batch "+j+" train="+result.getValue());//+" test="+testError);
			}
			
			AutoencoderFct autoencoderFct = new AutoencoderFct(brParams, testData, conf);
			double testError = autoencoderFct.computeTestError();
			System.out.println("Epoch "+i+" test="+testError);
		}

		return AutoencoderLinAlgebra.getFilters(params);
	}

	private AutoencoderParams initializeWeights(){
		return AutoencoderLinAlgebra.initialize(num_input,num_hidden);
	}

	private void split(JavaRDD<Vector> data){
		JavaRDD<Vector>[] splits = data.randomSplit(new double[]{0.8,0.2}, System.currentTimeMillis());
		trainData = splits[0].cache();
		testData  = splits[1].cache();
	}

	

	

	

	

	

}
