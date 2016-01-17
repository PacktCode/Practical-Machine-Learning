/**
* Practical Machine learning
* Random Forest Example
* Chapter 05
* @author sunilag
*/
package com.packt.pml.mahout.randomforest;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class RandomForestTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public RandomForestTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( RandomForestTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testRandomForest()
    {
        assertTrue( true );
    }
}
