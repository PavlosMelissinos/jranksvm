package mlpms;

import static org.junit.Assert.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import mulan.core.MulanException;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Test;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

public class RankSVMTest {

	@Test
	public void testBuildInternalSetup() throws InvalidDataFormatException, Exception{
	    // Input Data
		MultiLabelInstances trainingSet;
		trainingSet = new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");
		
		// Expected Results 
		// Size_alpha
		MatFileReader reader1 = new MatFileReader("data/size_alpha.mat");
		MLDouble size_alpha_mat = (MLDouble)reader1.getMLArray("size_alpha");
		BlockRealMatrix sizeAlphaMat = new BlockRealMatrix(size_alpha_mat.getArray());
		// Label_size
		MatFileReader reader2 = new MatFileReader("data/Label_size.mat");
		MLDouble label_size_mat = (MLDouble)reader2.getMLArray("Label_size");
		BlockRealMatrix labelSizeMat = new BlockRealMatrix(label_size_mat.getArray());
		//Label
		MatFileReader reader3 = new MatFileReader("data/label.mat");
		//MLCell label_mat = (MLCell) reader3.getMLArray("Label");
		MLCell label_mat = (MLCell) reader3.getMLArray("Label");
		ArrayList<MLArray> labelMat = new ArrayList<MLArray>(label_mat.cells());
		//ArrayList<ML> labelMat = new ArrayList<MLDouble>(((MLDouble) label_mat).getArray());
	
	    // MyClass is tested
	    RankSVM tester = new  RankSVM();
	    HashMap<String, Object>  results = tester.setup(trainingSet);
		results  = tester.setup(trainingSet);
		double delta = 0.0001;
		//Size_alpha 
		ArrayRealVector sizeAlpha = (ArrayRealVector) results.get("sizeAlpha");
        assertArrayEquals(sizeAlphaMat.getRow(0), sizeAlpha.toArray(), delta);
		System.out.println("OK");
		// Label_size
		ArrayRealVector labelSize = (ArrayRealVector) results.get("labelSize");
		assertArrayEquals(labelSizeMat.getRow(0), labelSize.toArray(), delta);
		System.out.println("OK");
		//Label
		ArrayList<ArrayRealVector> label = (ArrayList<ArrayRealVector>) results.get("Label");
		assertArrayEquals(labelMat.toArray() , label.toArray());
		System.out.println("OK");
	}

	
	
	/*public void testBuildInterna() throws Exception {
	    fail("Not yet implemented");
	}*/

}
