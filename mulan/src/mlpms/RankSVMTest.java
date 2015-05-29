package mlpms;

import static org.junit.Assert.assertArrayEquals;

import java.util.ArrayList;
import java.util.HashMap;

import mlpms.RankSVM.KernelType;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Test;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLStructure;

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

	@Test
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
		HashMap<String, Object> results = tester.setup(trainingSet);
		BlockRealMatrix SVs =  tester.getSVs();
		BlockRealMatrix kernelRBF = tester.KernelsSetup(trainingSet,  SVs );
		double delta = 0.0001;
		//assertArrayEquals(kernelRBF.getData(), kernelRBFMat.getData());
		for (int i = 0; i < kernelRBF.getColumnDimension(); i++){
		assertArrayEquals(kernelRBF.getRow(i), kernelRBFMat.getRow(i), delta);
		System.out.println("OK");
		}
		System.out.println("OK");

	}
	
	@Test
	public void testMakePredictionInternal() throws IllegalArgumentException, Exception{
		
		MultiLabelInstances trainingSet =
				new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");
		MultiLabelInstances testingSet =
				new MultiLabelInstances("data/yeast-test.arff", "data/yeast.xml");
		
		//MatFileReader reader1 = new MatFileReader("data/matlabWorkspace.mat");
		MatFileReader reader1 = new MatFileReader("data/matlabWorkspaceAll.mat");
		
		MLStructure svmML = (MLStructure) reader1.getMLArray("svm");
		MLChar type = (MLChar) svmML.getField("type");
		String typeStr = type.contentToString();
		KernelType kType = KernelType.valueOf(typeStr);
		
		MLDouble costML = (MLDouble) svmML.getField("cost", 0);
		double cost = new BlockRealMatrix(costML.getArray()).getEntry(0, 0);
		
		MLDouble weightsML = (MLDouble) reader1.getMLArray("Weights");
		BlockRealMatrix weights = new BlockRealMatrix(weightsML.getArray());

		MLDouble biasML = (MLDouble) reader1.getMLArray("Bias");
		ArrayRealVector bias = (ArrayRealVector) new BlockRealMatrix(biasML.getArray()).getRowVector(0);
		
		MLDouble SVsML = (MLDouble) reader1.getMLArray("SVs");
		BlockRealMatrix SVs = new BlockRealMatrix(weightsML.getArray());
		
		MLDouble weightsSizePreML = (MLDouble) reader1.getMLArray("Weights_sizepre");
		ArrayRealVector weightsSizePre = 
				(ArrayRealVector) new BlockRealMatrix(weightsSizePreML.getArray()).getRowVector(0);
		
		MLDouble biasSizePreML = (MLDouble) reader1.getMLArray("Bias_sizepre");
		double biasSizePre = new BlockRealMatrix(biasSizePreML.getArray()).getEntry(0, 0);

		RankSVM classifier = new RankSVM(weights, bias, SVs, weightsSizePre, biasSizePre);
		//classifier.setKernelOptions(kType, cost, gamma, coefficient, degree);
		//classifier
		Evaluator eval = new Evaluator();
		Evaluation results = eval.evaluate(classifier, testingSet, trainingSet);
	}



	/*public void testBuildInterna() throws Exception {
    fail("Not yet implemented");
}*/

}