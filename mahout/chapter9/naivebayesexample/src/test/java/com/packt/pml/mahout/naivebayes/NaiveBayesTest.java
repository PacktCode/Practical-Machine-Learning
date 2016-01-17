package com.packt.pml.mahout.naivebayes;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class NaiveBayesTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public NaiveBayesTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( NaiveBayesTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testNaiveBayes()
    {
        assertTrue( true );
    }
}
