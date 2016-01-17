/**
* Practical Machine Learning
* Aoocition rule based learning - FP growth example
* Chapter 07
**/
package com.packt.pml.mahout.fpgrowth;

import org.apache.mahout.fpm.pfpgrowth.fpgrowth.*;

import java.io.IOException;
import java.util.*;

import org.apache.mahout.common.iterator.*;
import org.apache.mahout.fpm.pfpgrowth.convertors.*;
import org.apache.mahout.fpm.pfpgrowth.convertors.integer.*;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.*;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.*;

import org.apache.hadoop.io.Text;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class FrequentPatternMiningFP {
    
    OutputCollector<String, List<Pair<List<String>, Long>>> output;
    Collection<Pair<String, Long>> frequencies;
    StatusUpdater updater;
    
    ArrayList<String> items;
    LinkedList<Pair<List<String>, Long>> transactions = new LinkedList<Pair<List<String>, Long>>();
    FPGrowth<String> fps = new FPGrowth<String>();
    
    long minSupport = 1L;
    int k = 50;

    
    String cvsFileName = "/data/fpgrowth/marketbasket.csv";
    
    private static final Logger log = LoggerFactory.getLogger(FrequentPatternMiningJava.class);    
    
    public static void main(String[] args) 
    {
        FrequentPatternMiningJava fpmj = new FrequentPatternMiningJava();
        fpmj.readItemsName();     
        fpmj.readTransactions();
        fpmj.findFrequencies();
        fpmj.createOutput();        
        fpmj.initUpdater();               
        fpmj.performPatternMining();
        fpmj.outputResults();
    } 
    
    private void initUpdater()
    {
        updater = new StatusUpdater() {
          public void update(String status) {
            log.info("updater :" + status);
          }
        };        
    }

    private void readItemsName() {
        
        if (items != null) 
            items.clear();
        log.info("adding items name");
            
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";
        
        items = new ArrayList<String>();

        try {

                br = new BufferedReader(new FileReader(cvsFileName));
                line = br.readLine();
                String[] itemss = line.split(",");
                
                for(int i=0;i<itemss.length;i++)
                {
                    //log.info("adding item " + itemss[i]);
                    
                    items.add(itemss[i]);
                }

                        

        } catch (FileNotFoundException e) {
                e.printStackTrace();
        } catch (IOException e) {
                e.printStackTrace();
        } finally {
                if (br != null) {
                        try {
                                br.close();
                        } catch (IOException e) {
                                e.printStackTrace();
                        }
                }
        }        
        
        
    }

    private void readTransactions() {
        
        if (transactions != null) 
            transactions.clear();
        log.info("adding transactions");
            
        BufferedReader br = null;
        String line = "";
        
        try {

                br = new BufferedReader(new FileReader(cvsFileName));
                line = br.readLine();                                                
                
                int j = 0;
                while ((line = br.readLine()) != null) { 
		        // use comma as separator
			String[] itemsintransaction = line.split(",");
                        ArrayList<String> ar = new ArrayList<String>();
                        for (int i = 0; i <  itemsintransaction.length; i++)
                        {
                            if (Integer.parseInt(itemsintransaction[i]) > 0 )
                            {
                               ar.add(items.get(i));
                            }                            
                        } 
                        if (ar.size() > 0)
                        {
                            log.info("adding a transaction of " + ar.toString());
                            transactions.add( new Pair(ar,1L) );                            
                        }    
                        j++;
  //                      if (j> 100 )
//                            System.exit(0);
		}                

        } catch (FileNotFoundException e) {
                log.error(e.getMessage());
        } catch (IOException e) {
                log.error(e.getMessage());
        } finally {
                if (br != null) {
                        try {
                                br.close();
                        } catch (IOException e) {
                                log.error(e.getMessage());
                        }
                }
        }        
        
        
    }

    private void findFrequencies() {
        frequencies = fps.generateFList(transactions.iterator(), (int) minSupport);
        
        //for (<String, Long> frequency :frequencies) 
        for (Pair<String, Long> frequency : frequencies)
        {
            log.info("frequency of item : " + frequency.toString() + " up to " + transactions.size()  );    
        }
        
        
        
    }

    private void performPatternMining() {
        try {
          fps.generateTopKFrequentPatterns(            
            transactions.iterator(), // use a "fresh" iterator
            frequencies, 
            minSupport, 
            k, 
            null, 
            output, 
            updater);
        } catch (Exception e) {
          e.printStackTrace();
        }        
    }

    private  void createOutput() {
       output = new OutputCollector<String, List<Pair<List<String>, Long>>>() {
           
          @Override
          public void collect(String x1, List<Pair<List<String>, Long>> listPair) throws IOException {
            StringBuffer sb = new StringBuffer();
            sb.append(x1 + ":");
            for (Pair<List<String>, Long> pair : listPair) {
              sb.append("[");
              String sep = "";
              for (String item : pair.getFirst()) {
                sb.append(item + sep);
                sep = ", ";
              }
              sb.append("]:" + pair.getSecond());
            }
            log.info(" createOutput " + sb.toString());
          }
        };
    }

    private void outputResults() {
        
    }
    
    
}
