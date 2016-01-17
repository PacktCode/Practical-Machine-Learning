/**
* Practical Machine learning
* Logistic Regression Example
* Chapter 10
*/
package com.packt.pml.mahout.logreg;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class LogisticRegressionTest 
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public LogisticRegressionTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( LogisticRegressionTest.class );
    }

    /**
     * Rigourous Test :-)
     */
    public void testLogisticRegression()
    {
        assertTrue( true );
    }
}
