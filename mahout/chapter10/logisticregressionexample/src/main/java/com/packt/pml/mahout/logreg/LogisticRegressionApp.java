/**
* Practical Machine learning
* Logistic Regression Example
* Chapter 10
*/
package com.packt.pml.mahout.logreg;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;


/**
 * Hello world!
 *
 */
public class LogisticRegressionApp 
{
    public static void main( String[] args ) throws FileNotFoundException, IOException
    {
            
        CSVReader reader = new CSVReader(new FileReader("$WORK_DIR/train/train.csv"));

        String [] nextLine;
        String [] previousLine;
        String [] headernew = new String [reader.readNext().length + 1];  
        
        CSVWriter writer = new CSVWriter(new FileWriter("$WORK_DIR/train/final.csv"), ',');  
        
        nextLine = reader.readNext();
        
        for (int i = 0; i < nextLine.length;i++)
        {
            headernew[i] = nextLine[i];
        }
        
        headernew[headernew.length-1] = "action";
        writer.writeNext(headernew); 
        
        previousLine = reader.readNext();
           
        
        while ((nextLine = reader.readNext()) != null) {
            // nextLine[] is an array of values from the line
            System.out.println(nextLine[0] + nextLine[1] + "etc...");
            headernew = new String [nextLine.length + 1];
            
            for (int i = 0; i < headernew.length-1;i++)
            {
                headernew[i] = nextLine[i];
            }            
            
            if (
                    Double.parseDouble(previousLine[4]) < Double.parseDouble(nextLine[4])
                )
            {
                    headernew[headernew.length] = "SELL";
            } else {
                headernew[headernew.length] = "BUY";
            }
            
            writer.writeNext(headernew);
            
            previousLine = nextLine;
            
            
        }
        
        reader.close();
        writer.close();

    }
}
