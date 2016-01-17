/**
* Practical Machine learning
* Logistic Regression Example
* Chapter 10
*/
package com.packt.pml.mahout.logreg;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Reader;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticModelParameters;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression.Wrapper;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 *
 * @author sunila-gollapudi
 */
public class LogisticRegressionBase {
    public static void main(String[] args) throws IOException 
    {
        String inputFile = "/mnt/new/logistic/train/google.csv";
        String outputFile = "/mnt/new/logistic/model/modelfromcode";
        
        AdaptiveLogisticModelParameters lmp = new AdaptiveLogisticModelParameters();
        int passes = 50;
        boolean showperf;
        int skipperfnum = 99;
        AdaptiveLogisticRegression model;        

        CsvRecordFactory csv = lmp.getCsvRecordFactory();
        model = lmp.createAdaptiveLogisticRegression();
        State<Wrapper, CrossFoldLearner> best;
        CrossFoldLearner learner = null;
        
        int k = 0;
        
        for (int pass = 0; pass < passes; pass++) {
                BufferedReader in = new BufferedReader(new FileReader(inputFile));

                
                csv.firstLine(in.readLine());

                String line = in.readLine();
                int lineCount = 2;
                while (line != null) {
                  
                  Vector input = new RandomAccessSparseVector(lmp.getNumFeatures());
                  int targetValue = csv.processLine(line, input);

                  // update model
                  model.train(targetValue, input);
                  k++;

                  line = in.readLine();
                  lineCount++;
                }
                in.close();
              }

            best = model.getBest();
            if (best != null) {
              learner = best.getPayload().getLearner();
            }


            OutputStream modelOutput = new FileOutputStream(outputFile);
            try {
              lmp.saveTo(modelOutput);
            } finally {
              modelOutput.close();
            }
    }
}
