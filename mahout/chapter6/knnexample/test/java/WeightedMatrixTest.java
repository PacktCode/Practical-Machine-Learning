// Practical Machine learning
// K Nearest Neighbor example 
// Chapter 6

package test.java;

import main.java.KNearestNeighbor;

import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class WeightedMatrixTest {

	public static void main(String[] args) {
		double testArray[][] = {{1,2,3,4,5,6},{1,2,3,4,5,7},{2,4,6,8,10,12},{2,3,4,5,6,7},{5,5,5,5,5,5}};
		System.out.println(testArray.length);
		Vector[] vectors = new Vector[5];

		for (int i = 0; i < 5; i++) {
			vectors[i] = Vectors.dense(testArray[i]);
			System.out.println("Vector " + i +": " + vectors[i].toArray()[0] + " " + vectors[i].toArray()[1] + " " + vectors[i].toArray()[2] + " " + vectors[i].toArray()[3] + " " + vectors[i].toArray()[4] + " " + vectors[i].toArray()[5] + " ");
		}
		KNearestNeighbor w = new KNearestNeighbor(vectors, 3,0,1,1);
		Matrix m = w.getWeightedMatrix();
		System.out.println("Matrix:");
		int size = m.numCols();
		double[] mA = m.toArray();
		for(int i=0; i<size; i++){
			for(int j=0; j<size; j++){
				System.out.print(mA[i + j*size] + " ");
			}
			System.out.print("\n");
		}

    }
}


