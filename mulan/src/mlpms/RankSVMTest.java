package mlpms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import junit.framework.Assert;
import mlpms.RankSVM.KernelType;
import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.junit.Assert;
import org.junit.Test;

import weka.core.Instance;

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
		tester.setKernelOptions(KernelType.RBF, 1, 1, 1, 1);
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
	public void testBuildInternal() throws InvalidDataFormatException, Exception{
		// Input Data
		MultiLabelInstances trainingSet;
		trainingSet = new MultiLabelInstances("data/yeast-train-10percent.arff", "data/yeast.xml");

		// Expected Results 
		// Size_alpha
		MatFileReader reader = new MatFileReader("data/matlab10Percent.mat");
		// Kernel
		MLDouble kernel_rbf_mat = (MLDouble)reader.getMLArray("kernel");
		BlockRealMatrix kernelRBFMat = new BlockRealMatrix(kernel_rbf_mat.getArray());
        // SVs
		MLDouble SVs_mat = (MLDouble)reader.getMLArray("SVs");
		BlockRealMatrix SVsMat = new BlockRealMatrix(SVs_mat.getArray());
		// Size_alpha
		MLDouble size_alpha_mat = (MLDouble)reader.getMLArray("size_alpha");
		BlockRealMatrix sizeAlphaMat = new BlockRealMatrix(size_alpha_mat.getArray());
		// Label size
		MLDouble label_size_mat = (MLDouble)reader.getMLArray("Label_size");
		BlockRealMatrix labelSizeMat = new BlockRealMatrix(label_size_mat.getArray());
		// Label
		MLCell label_mat = (MLCell) reader.getMLArray("Label");
		ArrayList<MLArray> labelMat = new ArrayList<MLArray>(label_mat.cells());
		// Not Label
		MLCell not_label_mat = (MLCell) reader.getMLArray("not_Label");
		ArrayList<MLArray> notLabelMat = new ArrayList<MLArray>(not_label_mat.cells());
        // Gradient
		MLDouble gradient_mat = (MLDouble)reader.getMLArray("gradient");
		BlockRealMatrix gradientMat = new BlockRealMatrix(gradient_mat.getArray());
        // Bias
		MLDouble bias_mat = (MLDouble)reader.getMLArray("Bias");
		BlockRealMatrix biasMat = new BlockRealMatrix(bias_mat.getArray());
        // Beta
		MLDouble beta_mat = (MLDouble)reader.getMLArray("Beta");
		BlockRealMatrix betaMat = new BlockRealMatrix(beta_mat.getArray());
        // weightsSizePre
		MLDouble weightsSizePre_mat = (MLDouble)reader.getMLArray("Weights_sizepre");
		BlockRealMatrix weightsSizePreMat = new BlockRealMatrix(weightsSizePre_mat.getArray());
        // biasSizePre
		MLDouble biasSizePreML = (MLDouble) reader.getMLArray("Bias_sizepre");
		double biasSizePre = new BlockRealMatrix(biasSizePreML.getArray()).getEntry(0, 0);
        // weights
		MLDouble weights_mat = (MLDouble) reader.getMLArray("Weights");
		BlockRealMatrix weightsMat = new BlockRealMatrix(weights_mat.getArray());
		System.out.println("OK");

		// MyClass is tested
		RankSVM tester = new  RankSVM();
		tester.setKernelOptions(KernelType.RBF, 1, 1, 1, 1);
		tester.build(trainingSet);
		ArrayList<ArrayRealVector> labelNew = new ArrayList<ArrayRealVector>();
		ArrayList<ArrayRealVector> notLabelNew = new ArrayList<ArrayRealVector>();
		for (int i = 0; i < labelSizeMat.getColumnDimension(); i++){
			MLArray temp1 = labelMat.get(i);
			BlockRealMatrix temp2 = new BlockRealMatrix(((MLDouble) temp1).getArray());  // Because Matlab starts indexing from 1 and java from 0.
			BlockRealMatrix temp3 = temp2.scalarAdd(-1);
			labelNew.add(i, (ArrayRealVector)temp3.getRowVector(0));
			MLArray temp4 = notLabelMat.get(i);
			BlockRealMatrix temp5 = new BlockRealMatrix(((MLDouble) temp4).getArray());  // Because Matlab starts indexing from 1 and java from 0.
			BlockRealMatrix temp6 = temp5.scalarAdd(-1);
			notLabelNew.add(i, (ArrayRealVector)temp6.getRowVector(0));
			System.out.println("OK");
			//assertArrayEquals(temp3.getRow(0), label2.toArray(),delta);	
		}
		ArrayRealVector gradient = tester.findAlpha((ArrayRealVector)sizeAlphaMat.getRowVector(0), (ArrayRealVector)labelSizeMat.getRowVector(0), labelNew, notLabelNew, kernelRBFMat);
		tester.computeBias((ArrayRealVector) labelSizeMat.getRowVector(0), (ArrayRealVector)sizeAlphaMat.getRowVector(0), labelNew , notLabelNew, gradient);
		tester.computeSizePredictor(betaMat, (ArrayRealVector)biasMat.getRowVector(0), kernelRBFMat, labelNew, notLabelNew);
		BlockRealMatrix weights= tester.getweights(); 
		double delta = 0.0001;
		// Gradient
		assertArrayEquals((double [])gradientMat.getRow(0), gradient.toArray(), delta);
		// Bias
		assertArrayEquals((double [])biasMat.getRow(0), tester.getBias().toArray(), delta);
		// weightsSizePre
		assertArrayEquals((double [])weightsSizePreMat.getRow(0), tester.getweightsbiasSizePre().toArray(), delta);
		// biasSizePre
		assertEquals(biasSizePre, tester.getbiasSizePre(), delta);
		// Weights
		for (int i = 0; i < weightsMat.getRowDimension(); i++){
		assertArrayEquals(weightsMat.getRow(i), weights.getRow(i), delta);
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

		MatFileReader reader1 = new MatFileReader("data/matlabPredict.mat");
		
		//MLStructure svmML = (MLStructure) reader1.getMLArray("svm");
		//MLChar type = (MLChar) svmML.getField("type");
		//String typeStr = type.contentToString();
		//KernelType kType = KernelType.valueOf(typeStr);
		
		//MLDouble costML = (MLDouble) svmML.getField("cost", 0);
		//double cost = new BlockRealMatrix(costML.getArray()).getEntry(0, 0);

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

		//RankSVM classifier = new RankSVM(weights, bias, SVs, weightsSizePre, biasSizePre);
		//classifier.setKernelOptions(kType, cost, gamma, coefficient, degree);
		RankSVM classifier = new RankSVM();
		classifier.build(trainingSet);
		
		for (int i = 0; i < testingSet.getNumInstances(); i++){
			Instance instance = testingSet.getDataSet().instance(i);
			MultiLabelOutput s = classifier.makePrediction(instance);
		}
		Evaluator eval = new Evaluator();
		ArrayList<Measure> measures = new ArrayList<Measure>();
		measures.add(new HammingLoss());
		measures.add(new RankingLoss());
		measures.add(new OneError());
		measures.add(new Coverage());
		measures.add(new AveragePrecision());

		//Evaluation results = eval.evaluate(classifier, testingSet, trainingSet);
		Evaluation results = eval.evaluate(classifier, testingSet, measures);
		List<Measure> measuresOut = results.getMeasures();
		
		MatFileReader reader2 = new MatFileReader("data/matlabPredictOutput.mat");
		
		MLDouble hLossML = (MLDouble) reader1.getMLArray("HammingLoss");
		double hammingLoss = new BlockRealMatrix(hLossML.getArray()).getEntry(0, 0);
		
		MLDouble rLossML = (MLDouble) reader1.getMLArray("RankingLoss");
		double rankingLoss = new BlockRealMatrix(rLossML.getArray()).getEntry(0, 0);
		
		MLDouble oneErrorML = (MLDouble) reader1.getMLArray("OneError");
		double oneError = new BlockRealMatrix(oneErrorML.getArray()).getEntry(0, 0);
		
		MLDouble coverageML = (MLDouble) reader1.getMLArray("Coverage");
		double coverage = new BlockRealMatrix(coverageML.getArray()).getEntry(0, 0);
		
		MLDouble averagePrecisionML = (MLDouble) reader1.getMLArray("Average_Precision");
		double averagePrecision = new BlockRealMatrix(averagePrecisionML.getArray()).getEntry(0, 0);

		MLDouble outputsML = (MLDouble) reader1.getMLArray("Outputs");
		double outputs = new BlockRealMatrix(outputsML.getArray()).getEntry(0, 0);

		MLDouble thresholdML = (MLDouble) reader1.getMLArray("Threshold");
		double threshold = new BlockRealMatrix(thresholdML.getArray()).getEntry(0, 0);

		MLDouble preLabelsML = (MLDouble) reader1.getMLArray("Pre_Labels");
		double preLabels = new BlockRealMatrix(preLabelsML.getArray()).getEntry(0, 0);
		
		Assert.assertNotNull(results);

		//Assert.assertEquals(results.toString());
	}



	/*public void testBuildInterna() throws Exception {
    fail("Not yet implemented");
}*/

}