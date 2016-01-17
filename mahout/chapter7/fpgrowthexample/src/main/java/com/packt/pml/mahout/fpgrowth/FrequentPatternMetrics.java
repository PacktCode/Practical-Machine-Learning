/**
* Practical Machine Learning
* Aoocition rule based learning - FP growth example
* Chapter 07
**/
package com.packt.pml.mahout.fpgrowth;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FrequentPatternMetrics {
    
    String  FREQUENCY_ITEM_LIST = "";
    String  FREQUENCY_ITEM_PATTERNS = "";
    Configuration configuration;
    FileSystem fs;
    Reader rd;
    private static final Logger log = LoggerFactory.getLogger(FrequentPatternMetrics.class);    
    double transactionCount;
    double minSupport;
    double minConfidence;
    Map<Integer, Long> frequency;
    
    
    public  FrequentPatternMetrics() 
    {
        
        FREQUENCY_ITEM_LIST = "";
        FREQUENCY_ITEM_PATTERNS = "";
        configuration = new Configuration();
    
        transactionCount = 0;
        minSupport = 0;
        minConfidence = 0 ;
        frequency = new HashMap<Integer, Long>();
        
    }
    
    public void Init() throws IOException
    {
        log.info("init process");
        fs = FileSystem.get(configuration);
        FREQUENCY_ITEM_LIST = "/data/fpgrowth/fList";
        FREQUENCY_ITEM_PATTERNS = "/data/fpgrowth/frequentpatterns/";
        GetTransactionCount();
        minSupport = 0.04;
        minConfidence = 0.4;
                
    }
    
    private void GetTransactionCount() throws IOException 
    {        
        LineNumberReader reader  = new LineNumberReader(new FileReader("/data/fpgrowth/retail.dat"));
        String lineRead = "";
        while ((lineRead = reader.readLine()) != null) {}
        transactionCount =  reader.getLineNumber(); 
        reader.close();    
    }
    
    public String get_FREQUENCY_ITEM_LIST()
    {
        return this.FREQUENCY_ITEM_LIST;
    }
    
    public void log(Object o)
    {
        System.out.println(o);
    }
    
    public void set_FREQUENCY_ITEM_LIST(String value)
    {
        this.FREQUENCY_ITEM_LIST = value;
    }
    
    public String get_FREQUENCY_ITEM_PATTERNS()
    {
        return this.FREQUENCY_ITEM_PATTERNS;
    }
    
    public void set_FREQUENCY_ITEM_PATTERNS(String value)
    {
        this.FREQUENCY_ITEM_PATTERNS = value;
    }
    
    public static void main(String[] argv) throws IOException
    {
        FrequentPatternMetrics fpm = new FrequentPatternMetrics();
        
        fpm.set_FREQUENCY_ITEM_LIST("/data/fpgrowth/fList");
        fpm.set_FREQUENCY_ITEM_PATTERNS("/data/fpgrowth/frequentpatterns/");
        
                
        fpm.Init();
        //fpm.readFrequency();
        
        fpm.readFrequentPatterns();        
        //fpm.outPutPatterns();
        
        
        
        
    }

    private void ReadFrequencies() throws IOException {
        
        rd = new SequenceFile.Reader(fs, new Path(FREQUENCY_ITEM_LIST), this.configuration);
        
        Text key = new Text();
        LongWritable value = new LongWritable();
        
        while(rd.next(key, value)) {
            log("find key " + key.toString() + " with value : " + value.get());
            frequency.put(Integer.parseInt(key.toString()), value.get());
        }        
    }

    private void readFrequentPatterns() throws IOException {
        rd = new SequenceFile.Reader(fs,new Path(this.FREQUENCY_ITEM_PATTERNS), configuration);
        Text key = new Text();
        TopKStringPatterns value = new TopKStringPatterns();
 
        while(rd.next(key, value)) {
            long firstFrequencyItem = -1;
            String firstItemId = null;
            List<Pair<List<String>,Long>> patterns = value.getPatterns();
            int i = 0;
            for(Pair<List<String>,Long> pair: patterns) {
                List itemList = pair.getFirst();
                Long occurrence = pair.getSecond();
                if (i == 0) {
                    firstFrequencyItem = occurrence;
                    firstItemId = itemList.get(0).toString();
                } else {
                    double support = (double)occurrence / transactionCount;
                    double confidence = (double)occurrence / firstFrequencyItem;
                    
                    List listWithoutFirstItem = new ArrayList();
                    for(Object itemId: itemList) {
                        if (!itemId.equals(firstItemId)) {
                            listWithoutFirstItem.add(itemId);
                        }
                    }
                    
                    long otherItemOccurrence = frequency.get(0);
                    double lift = (double)occurrence / (firstFrequencyItem * otherItemOccurrence);
                    double conviction = (1.0 - (double)otherItemOccurrence / transactionCount) / (1.0 - confidence);
                    i++;
            }
        }
        rd.close();        
    }


    
    }
}
