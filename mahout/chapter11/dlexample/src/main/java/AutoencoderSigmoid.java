// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

public class AutoencoderSigmoid {
	
	
	//Thread safe singleton class since Spark is run one thread per Partition
	
	private static volatile AutoencoderSigmoid instance = null;
	
	
	
	private AutoencoderSigmoid(){
			
	}
	
	public static AutoencoderSigmoid getInstance(){
		
		if (instance == null){
			synchronized (AutoencoderSigmoid.class) {
				if (instance == null){
					instance = new AutoencoderSigmoid();
				}
			}
		}
		return instance;
	}

	
	public static double getValue(double x){
		int i = (int) Math.round(x*100);
		return values[i];
		//alternatively make a 3 point average
	}
	//to complete with generated values
	private static double[] values = new double[]{-1.0,0.0,1.0};
}
