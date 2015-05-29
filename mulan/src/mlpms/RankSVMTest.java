package mlpms;

import static org.junit.Assert.*;

import java.awt.List;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import junit.framework.Assert;
import mulan.core.MulanException;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

public class RankSVMTest {

	@SuppressWarnings("unchecked")
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

		// Not Label
		MatFileReader reader4 = new MatFileReader("data/not_label.mat");
		//MLCell label_mat = (MLCell) reader3.getMLArray("Label");
		MLCell not_label_mat = (MLCell) reader4.getMLArray("not_Label");
		ArrayList<MLArray> notLabelMat = new ArrayList<MLArray>(not_label_mat.cells());

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
		for (int i = 0; i < label.size(); i++){
			MLArray temp1 = labelMat.get(i);
			BlockRealMatrix temp2 = new BlockRealMatrix(((MLDouble) temp1).getArray());  // Because Matlab starts indexing from 1 and java from 0.
			BlockRealMatrix temp3 = temp2.scalarAdd(-1);
			ArrayRealVector label2 =  (ArrayRealVector) label.get(i);
			System.out.println("OK");
			assertArrayEquals(temp3.getRow(0), label2.toArray(),delta);	
		}

		//Not Label
		ArrayList<ArrayRealVector> notLabel= (ArrayList<ArrayRealVector>) results.get("notLabel");
		for (int i = 0; i < notLabel.size(); i++){
			MLArray temp1 = notLabelMat.get(i);
			BlockRealMatrix temp2 = new BlockRealMatrix(((MLDouble) temp1).getArray());
			BlockRealMatrix temp3 = temp2.scalarAdd(-1); // Because Matlab starts indexing from 1 and java from 0.
			ArrayRealVector label2 =  (ArrayRealVector) notLabel.get(i);
			System.out.println("OK");
			assertArrayEquals(temp3.getRow(0), label2.toArray(),delta);	
		}
		System.out.println("OK");
	}



	/*public void testBuildInterna() throws Exception {
	    fail("Not yet implemented");
	}*/


	@SuppressWarnings("unchecked")
	@Test
	public void testBuildInternalgetSVs() throws InvalidDataFormatException, Exception{
		// Input Data
		MultiLabelInstances trainingSet;
		trainingSet = new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");

		// Expected Results 
		// SVs
		MatFileReader reader1 = new MatFileReader("data/SVs.mat");
		MLDouble SVs_mat = (MLDouble)reader1.getMLArray("SVs");
		BlockRealMatrix SVSMat = new BlockRealMatrix(SVs_mat.getArray());

		// MyClass is tested
		RankSVM tester = new  RankSVM();
		HashMap<String, Object>  results = tester.setup(trainingSet);
		BlockRealMatrix SVs = tester.getSVs();

		//Size_alpha 
		double delta = 0.0001;
		assertArrayEquals(SVSMat.getData(), SVs.getData());
		System.out.println("OK");

	}

/*	@Test
	public void testBuildInternalKernelsSetup() throws InvalidDataFormatException, Exception{
		// Input Data
		MultiLabelInstances trainingSet;
		trainingSet = new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");

		// Expected Results 
		// Size_alpha
		MatFileReader reader1 = new MatFileReader("data/kernelRBF.mat");
		MLDouble kernel_rbf_mat = (MLDouble)reader1.getMLArray("kernel");
		BlockRealMatrix kernelRBFMat = new BlockRealMatrix(kernel_rbf_mat.getArray());


		// MyClass is tested
		RankSVM tester = new  RankSVM();
		HashMap<String, Object>  results = tester.setup(trainingSet);
		//BlockRealMatrix kernelRBF = KernelsSetup(trainingSet, getSVs());

		results  = tester.setup(trainingSet);
		double delta = 0.0001;


	}
*/


	/*public void testBuildInterna() throws Exception {
    fail("Not yet implemented");
}*/

}