package com.packt.pml.mahout.naivebayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.Algorithm;
import org.apache.mahout.classifier.bayes.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.ClassifierContext;
import org.apache.mahout.classifier.bayes.Datastore;
import org.apache.mahout.classifier.bayes.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.TrainClassifier;
import org.apache.mahout.common.nlp.NGrams;


public class NaiveBayes 
{
    
    
    public static void main( String[] args ) throws InvalidDatastoreException
    {
        final BayesParameters params = new BayesParameters();
        params.setGramSize( 1 );
        params.set( "verbose", "true" );
        params.set( "classifierType", "bayes" );
        params.set( "defaultCat", "OTHER" );
        params.set( "encoding", "UTF-8" );
        params.set( "alpha_i", "1.0" );
        params.set( "dataSource", "hdfs" );
        params.set( "basePath", "/tmp/output" );
        
        try {
            Path input = new Path( "/tmp/input" );
            Path output = new Path( "/tmp/output" );
            
            TrainClassifier.trainNaiveBayes( input, output, params );

            Algorithm algorithm = new BayesAlgorithm();
            Datastore datastore = new InMemoryBayesDatastore( params );
            ClassifierContext classifier = new ClassifierContext( algorithm, datastore );
            classifier.initialize();

            final BufferedReader reader = new BufferedReader( new FileReader( args[ 0 ] ) );
            String entry = reader.readLine();

            while( entry != null ) {
                List< String > document = new NGrams( entry, 
                                Integer.parseInt( params.get( "gramSize" ) ) )
                                .generateNGramsWithoutLabel();

                ClassifierResult result = classifier.classifyDocument( 
                                 document.toArray( new String[ document.size() ] ), 
                                 params.get( "defaultCat" ) );          

                entry = reader.readLine();
            }
        } catch( final IOException ex ) {
         ex.printStackTrace();
        } catch( final InvalidDatastoreException ex ) {
         ex.printStackTrace();
        }        

    }
}
