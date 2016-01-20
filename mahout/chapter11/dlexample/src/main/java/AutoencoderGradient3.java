// Practical Machine learning
// Deep learning - Autoencoder example 
// Chapter 11

package main.java;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

public class AutoencoderGradient3 {
			//TO DO
			// 1 - regenerate serialID for Function2
			// 2 - Singleton Sigmoid function
			// 5 - replace faulty log code
			
			private Broadcast<AutoencoderParams> params;
			private JavaRDD<Vector> data;
			private double rho = 0.05;
			private double lambda = 0.001;
			private double beta = 6;

			public AutoencoderGradient3(Broadcast<AutoencoderParams> params, JavaRDD<Vector> data, AutoencoderConfig conf){
				this.params = params;
				this.data = data;
				rho = conf.getRho();
				lambda = conf.getLambda();
				beta = conf.getBeta();
			}
			
			public AutoencoderFctGrd getGradient(){
				

				FirstLayerActivation firstLayerActivation = new FirstLayerActivation(params);
				JavaRDD<Vector> a2 = data.mapPartitions(firstLayerActivation); 	
							
				ComputeRho computeRho = new ComputeRho();
				Vector rho_h =  a2.reduce(computeRho);
							
				long numSamples = data.count();
				
				double[] rhoH = rho_h.toArray();
				double[] sparsityArray = new double[rhoH.length];
				
				double KL = 0;
				double t1,t2,t3,t4;
				for (int i=0;i<rhoH.length;i++){
					
					//rhoH might be negative
					try{
						t1 = rho * Math.log(rho/rhoH[i]*numSamples);
					}
					catch (Exception e){
						t1 = 0;
					}
					
					//1 case would be rhoH = 1;
					try{
						t2 = (1-rho) * Math.log((1-rho)/(1-rhoH[i]/numSamples));
					}
					catch (Exception e){
						t2 = 0;
					}
					
					KL += t1+t2;
					
					try{
						t3 = -rho/rhoH[i]*numSamples; 
					}catch (Exception e){
						t3 = 0;
					}
					
					try{
						t4 = (1-rho)/(1-rhoH[i]/numSamples); 
					}catch (Exception e){
						t4 = 0;
					}
					sparsityArray[i] = beta*(t3+t4);
					
				}
				
				JavaSparkContext sc = JavaSparkContext.fromSparkContext(data.context());
				Broadcast<AutoencoderComputedParams> computedBroadcast = sc.broadcast(new AutoencoderComputedParams(numSamples, sparsityArray));
	
				double norm1 = Vectors.norm((Vector)new DenseVector(params.getValue().getW1().toArray()),2);
				double norm2 = Vectors.norm((Vector)new DenseVector(params.getValue().getW2().toArray()),2);

				Compute compute = new Compute(params, computedBroadcast);
				Sum  	sum     = new Sum();
				AutoencoderFctGrd partialFctGrad = data.mapPartitions(compute).reduce(sum);
				
				int num_input = params.getValue().getW1().numCols();
				int num_hidden = params.getValue().getW1().numRows();
				
				DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
						new double[num_hidden * num_input]);
				for (int i=0;i<num_hidden;i++){
					for (int j=0;j<num_input;j++){
						w1.update(i, j, lambda*params.getValue().getW1().apply(i, j)+partialFctGrad.getW1().apply(i, j));
					}
				}
				
				DenseMatrix w2 = new DenseMatrix(num_input,num_hidden , 
						new double[num_hidden * num_input]);
				for (int i=0;i<num_input;i++){
					for (int j=0;j<num_hidden;j++){
						w2.update(i, j, lambda*params.getValue().getW2().apply(i, j)+partialFctGrad.getW2().apply(i, j));
					}
				};
	
				return new AutoencoderFctGrd(w1,w2,partialFctGrad.getB1(),partialFctGrad.getB2(),
							partialFctGrad.getValue()+0.5*lambda*(norm1*norm1+norm2*norm2)+beta*KL);		
			}
			
			private static class FirstLayerActivation implements FlatMapFunction<Iterator<Vector>,Vector>{
				
			
				private Broadcast<AutoencoderParams> params;
				
				public FirstLayerActivation(Broadcast<AutoencoderParams> params) {
					this.params = params;
				}

				
				public Iterable<Vector> call(Iterator<Vector> arg0) throws Exception {
//					double[] b1V = params.getValue().getB1().toArray();
//					DenseVector result = new DenseVector(params.getValue().getB1().toArray());
//					BLAS.gemv(1.0, params.getValue().getW1(), (DenseVector) arg0._1, 0.0, result);
					double[] values = new double[params.getValue().getW1().numRows()];
					while(arg0.hasNext()){
					double[] newValue = AutoencoderLinAlgebra.MVM(params.getValue().getW1(), arg0.next());
					for(int i=0;i<values.length;i++){
						values[i] += AutoencoderLinAlgebra.sigmoid(newValue[i]+params.getValue().getB1().apply(i));
						//Avoid complex computation and use a Singleton class to get precomputed values
						//AutoencoderSigmoid.getInstance().getValue(values[i]);
					}
					}
					ArrayList<Vector> result = new ArrayList<Vector>();
					result.add(new DenseVector(values));
					return result;
				}
				
			}
			
			private static class ComputeRho implements Function2<Vector, Vector, Vector>{

				@Override
				public Vector call(Vector arg0,  Vector arg1) throws Exception {
//					DenseVector result = new DenseVector(arg0._1.toArray());
//					BLAS.axpy(1.0, arg1._1, result);
					return new DenseVector(AutoencoderLinAlgebra.VVA(arg0,arg1)); 
				}
				
			}
			
		
			private static class Compute implements FlatMapFunction<Iterator<Vector>,AutoencoderFctGrd>{
				private Broadcast<AutoencoderParams> params;
				private Broadcast<AutoencoderComputedParams> comp;
				
				public Compute(Broadcast<AutoencoderParams> params,Broadcast<AutoencoderComputedParams> comp){
					this.params = params;
					this.comp = comp;
				}

				@Override
				public Iterable<AutoencoderFctGrd> call( Iterator<Vector> arg0)
						throws Exception {
					
					int num_input = params.getValue().getW1().numCols();
					int num_hidden = params.getValue().getW1().numRows();
					long num_samples = comp.getValue().getNumSamples();
					
					double[][] grad_W2 = new double[num_input][num_hidden];
					double[][] grad_W1 = new double[num_hidden][num_input];
					double[]   grad_b1 = new double[num_hidden];
					double[]   grad_b2 = new double[num_input];
					double fctValue = 0;
					
					while(arg0.hasNext()){
					
					double[] x  = arg0.next().toArray();
					double[] a2A = AutoencoderLinAlgebra.MAM(params.getValue().getW1(), x);					
					for(int i=0;i<a2A.length;i++){
						a2A[i] = AutoencoderLinAlgebra.sigmoid(a2A[i]+params.getValue().getB1().apply(i));
						//Avoid complex computation and use a Singleton class to get precomputed values
						//AutoencoderSigmoid.getInstance().getValue(values[i]);
						
					}
					
										
					
					
					double[] delta3A = AutoencoderLinAlgebra.VAA(params.getValue().getB2(),
							AutoencoderLinAlgebra.MAM(params.getValue().getW2(),a2A)); 
					double[] b3A = new double[delta3A.length];
					for(int i=0;i<delta3A.length;i++){
						double sig = AutoencoderLinAlgebra.sigmoid(delta3A[i]);
							//Avoid complex computation and use a Singleton class to get precomputed values
						//AutoencoderSigmoid.getInstance().getValue(values[i]);
						double val = (x[i]-sig);
						fctValue += val*val;
						delta3A[i] = -val*(1-sig)*sig;	
						b3A[i] = delta3A[i]/num_samples;
					}
					
						
					double[] delta2A = AutoencoderLinAlgebra.AAA(comp.getValue().getSparsityArray(),AutoencoderLinAlgebra.MTAM(params.getValue().getW2(),delta3A));
					double[] b2A = new double[delta2A.length];
					for(int i=0;i<delta2A.length;i++){									
						delta2A[i] = delta2A[i]*a2A[i]*(1-a2A[i]);	
						b2A[i] = delta2A[i]/num_samples;
					}
					
					
					
					for (int j=0;j<num_hidden;j++){
					for (int i=0;i<num_input;i++){
						grad_W2[i][j]+=	delta3A[i]*a2A[j]/num_samples;					
						}
					}				

								
										
					for (int i=0;i<num_hidden;i++){
					for (int j=0;j<num_input;j++){
						grad_W1[i][j]+= delta2A[i]*x[j]/num_samples;					
						}
					}
					
					for(int i=0;i<num_hidden;i++){
						grad_b1[i] += b2A[i];
					}
					for(int i=0;i<num_input;i++){
						grad_b2[i] += b3A[i];
					}

					
					};
					
					DenseMatrix W2 = new DenseMatrix(num_input,num_hidden,  new double[num_hidden*num_input]);					
					for (int j=0;j<num_hidden;j++){
					for (int i=0;i<num_input;i++){
						W2.update(i, j,	grad_W2[i][j]);					
						}
					}				

								
					DenseMatrix W1 = new DenseMatrix( num_hidden,num_input, new double[num_hidden*num_input]);					
					for (int i=0;i<num_hidden;i++){
					for (int j=0;j<num_input;j++){
							W1.update(i, j, grad_W1[i][j]);					
						}
					}
					
					DenseVector b1 = new DenseVector(grad_b1);
					DenseVector b2  = new DenseVector(grad_b2);
					
					ArrayList<AutoencoderFctGrd> result = new ArrayList<AutoencoderFctGrd>();
					result.add(new AutoencoderFctGrd(W1,W2,b1,b2,0.5*fctValue/comp.getValue().getNumSamples()));
					return result;
				}
				
			}
			
			private static class Sum implements Function2<AutoencoderFctGrd, AutoencoderFctGrd, AutoencoderFctGrd>{

				@Override
				public AutoencoderFctGrd call(AutoencoderFctGrd arg0,
						AutoencoderFctGrd arg1) throws Exception {
					int num_input = arg0.getW1().numCols();
					int num_hidden = arg0.getW1().numRows();
					DenseMatrix w1 = new DenseMatrix(num_hidden, num_input, 
							new double[num_hidden * num_input]);
					for (int i=0;i<num_hidden;i++){
						for (int j=0;j<num_input;j++){
							w1.update(i, j, arg0.getW1().apply(i, j)+arg1.getW1().apply(i, j));
						}
					}
					
					DenseMatrix w2 = new DenseMatrix(num_input, num_hidden, 
							new double[num_hidden * num_input]);
					for (int i=0;i<num_input;i++){
						for (int j=0;j<num_hidden;j++){
							w2.update(i, j, arg0.getW2().apply(i, j)+arg1.getW2().apply(i, j));
						}
					}
					
					double[] b1A = new double[num_hidden];
					for (int i=0;i<num_hidden;i++){
						b1A[i] = arg0.getB1().apply(i)+arg1.getB1().apply(i);
					}
					
					DenseVector b1 = new DenseVector(b1A);
					
					double[] b2A = new double[num_input];
					for (int i=0;i<num_input;i++){
						b2A[i] = arg0.getB2().apply(i)+arg1.getB2().apply(i);
					}
					
					DenseVector b2 = new DenseVector(b2A);
			
					return new AutoencoderFctGrd(w1, w2, b1, b2, arg0.getValue()+arg1.getValue());
				}
				
			}
			
}
