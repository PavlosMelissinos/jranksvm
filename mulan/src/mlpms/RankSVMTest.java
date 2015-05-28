package mlpms;

import static org.junit.Assert.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import mulan.core.MulanException;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.junit.Test;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class RankSVMTest {

	@Test
	public void testBuildInternal() throws InvalidDataFormatException, Exception{
	    // Input Data
		MultiLabelInstances trainingSet;
		trainingSet = new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");
		
		// Expected Results 
		HashMap<String, Object> expResults = new HashMap<String, Object>();
		//MatFileReader reader = new MatFileReader("data/sizeAlphaArray.mat");
		MatFileReader reader = new MatFileReader("data/size_alpha.mat");
		//MLArray size_alpha_mat = reader.getMLArray("sizeAlphaArray");
		//MLArray size_alpha_mat = reader.getMLArray("target");
		MLArray size_alpha_mat =  reader.getMLArray("size_alpha");
		expResults.put("sizeAlpha", size_alpha_mat);
		//expResults.put("labelSize", labelSize);
		//expResults.put("Label", Label);
		//expResults.put("notLabel", notLabel);
		
	    // MyClass is tested
	    RankSVM tester = new  RankSVM();
	    HashMap<String, Object>  results = tester.setup(trainingSet);
		results  = tester.setup(trainingSet);
		assertSame(expResults.get(1), results.get(1));
	}

	
	/*public void testBuildInterna() throws Exception {
	    fail("Not yet implemented");
	}*/

}
