/** Practical Machine learning
* Ensemble learning
* Chapter 13
**/

package com.packt.pml.mahout.ensemble;

/*
 * A collection of utility functions for working with the ensemble
 * 
 */
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

class Utilities{
	HashMap<Long, ArrayList<Float>> hm;
	public Utilities() {
		// TODO Auto-generated constructor stub
		hm = new  HashMap<Long, ArrayList<Float>>();
	}
	
	public void insert(Long item, Float value){
		
		if(!hm.containsKey(item))
			hm.put(item, new ArrayList<Float>());
		hm.get(item).add(value);
	}
	
	public void show(){
		System.out.println(hm);
	}
	
	public HashMap<Long,Float> getAverage(){
		HashMap<Long,Float> result = new HashMap<Long,Float>();
		Set<Long> items = hm.keySet();
		Iterator<Long> it = items.iterator();
		float sum,avg;
		while(it.hasNext()){
			Long key = it.next();
			List<Float> values = hm.get(key);
			Iterator<Float> itv = values.iterator();
			sum=0;
			while(itv.hasNext())
				sum = sum + itv.next();
			avg = sum/values.size();
			result.put(key, new Float(avg));
		}
		
		return result;
	}
}
