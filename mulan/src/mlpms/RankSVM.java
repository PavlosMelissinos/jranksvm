package mlpms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <!-- globalinfo-start --> <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> <!-- technical-bibtex-end -->
 * 
 * @author pavlos
 *
 */

@SuppressWarnings("serial")
public class RankSVM extends MultiLabelLearnerBase {

	/** Train_dataset. */
	private MultiLabelInstances trainingSet;

	/** Train_data (only features). */
	private double[][] train_data;

	/** Train labels. */
	private double[][] train_target;

	private BlockRealMatrix SVs;

	private BlockRealMatrix target;

	private ArrayList<BlockRealMatrix> cValue;

	private ArrayRealVector alpha;

	private double cost;

	private double gamma;

	private double degree;

	private String type;

	private int[][] targetValues;

	/** Train labels. */
	// protected double[] m_class;
	private double[][] support_vectors;

	public void setCost() {
		this.cost = 1;
	}

	public void setGamma() {
		this.gamma = 1;
	}

	public void setDegree() {
		this.degree = 1;
	}

	public void setType() {
		this.type = "RBF";
	}

	public BlockRealMatrix getSVs() {
		return this.SVs;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		//PreprocessingStep1(trainingSet);
		setup(trainingSet);
		setKernelOptions("RBF", 1, 1, 0);
		KernelsSetup(trainingSet, getSVs());

		//findAlpha();
		//computeBias();
		//computeSizePredictor();
	}

	void setup(MultiLabelInstances trainingSet) {
		// Preprocessing - Initialize support vectors & targets (labels)

		int[] labelIndices = trainingSet.getLabelIndices();
		// ArrayList labelIndices = new ArrayList(Arrays.asList(tempLabels));
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();

		Instances insts = trainingSet.getDataSet();
		int numAttr = insts.numAttributes();
		int numFeatures = numAttr - numClass;

		SVs = new BlockRealMatrix(numTraining, numFeatures);
		target = new BlockRealMatrix(numTraining, numClass);
		int omitted = 0;
		for (int i = 0; i < numTraining; i++) {
			Instance inst = insts.get(i);
			double sumTemp = 0;
			ArrayRealVector SVRow = new ArrayRealVector(numFeatures);
			ArrayRealVector targetRow = new ArrayRealVector(numClass);
			int labelsChecked = 0;
			for (int attr = 0; attr < numAttr; attr++) {
				if (labelIndices[labelsChecked] == attr) {
					targetRow.setEntry(labelsChecked, inst.value(attr));
					sumTemp += inst.value(attr);
					labelsChecked++;
				} else
					SVRow.setEntry(attr - labelsChecked, inst.value(attr));
			}
			// filter out every instance that contains all or none of the labels
			// (uninformative)
			if ((sumTemp != numClass) && (sumTemp != -numClass)) {
				SVs.setRowVector(i-omitted, SVRow);
				target.setRowVector(i-omitted, targetRow);
			} else
				omitted++;
		}
		this.SVs.getSubMatrix(0, numTraining - omitted - 1, 0, numFeatures - 1);
		this.SVs = this.SVs.transpose(); //numInstances x numFeatures --> numFeatures x numInstances
		this.target.getSubMatrix(0, numTraining - omitted - 1, 0, numClass - 1);
	}

	private void PreprocessingStep1(MultiLabelInstances trainingSet) {

		int[] labelIndices = trainingSet.getLabelIndices();
		// dataset preprocessing
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();
		int numAttributes = trainingSet.getFeatureIndices().length;
		double[][] SVsInit = new double[numTraining][numAttributes];
		int[][] train_targetInit = new int[numTraining][numClass];

		int count = 0;
		int obs = 0;
		int ommited = 0;
		for (Instance inst : trainingSet.getDataSet()) {
			int[] labels = new int[labelIndices.length];
			for (int i = 0; i < labelIndices.length; i++) {
				labels[i] = (int) (inst.value(labelIndices[i]));
			}
			int sumTemp = IntStream.of(labels).sum();
			if ((sumTemp != numClass) && (sumTemp != -numClass)) {
				double[] tempMat = new double[numAttributes];
				System.arraycopy(inst.toDoubleArray(), 0, tempMat, 0,
						numAttributes);
				for (int k = 0; k < numAttributes; k++) {
					SVsInit[obs][k] = tempMat[k];
				}
				for (int l = 0; l < numLabels; l++) {
					train_targetInit[obs][l] = labels[l];
				}
				obs++;
			} else {
				ommited++;
			}
			count++;
		}
		double[][] SVs = new double[numTraining - ommited][numAttributes];
		int[][] train_target = new int[numTraining - ommited][numClass];
		for (int i = 0; i < numTraining - ommited; i++) {
			System.arraycopy(SVsInit[i], 0, SVs[i], 0, numAttributes);
			System.arraycopy(train_targetInit[i], 0, train_target[i], 0,
					numClass);
		}
		if (Arrays.deepEquals(SVsInit, SVs))
			System.out.println("Equal.");
		else
			System.out.println("Different.");
		if (Arrays.deepEquals(train_targetInit, train_target))
			System.out.println("Equal.");
		else
			System.out.println("Different.");
		System.out.println("OK");

		// Chunk2
		// int dim = SVs.length;
		int[] Label_size = new int[numTraining];
		int[] size_alpha = new int[numTraining];
		ArrayList<ArrayList<Integer>> Label = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> not_Label = new ArrayList<ArrayList<Integer>>();
		/* Initialize ArrayList of ArrayLists to have specific number of rows. */
		for (int i = 0; i < numTraining - ommited; i++) {
			Label.add(null);
			not_Label.add(null);
		}

		for (int i = 0; i < numTraining - ommited; i++) {
			ArrayList<Integer> train_target_temp = new ArrayList<Integer>();
			for (int j = 0; j < numClass; j++) {
				train_target_temp.add(train_target[i][j]); // temp label vector
			}
			Label_size[i] = train_target_temp.stream()
					.mapToInt(Integer::intValue).sum();
			size_alpha[i] = Label_size[i] * (numClass - Label_size[i]);

			ArrayList<Integer> LabelTemp = new ArrayList<Integer>();
			ArrayList<Integer> not_LabelTemp = new ArrayList<Integer>();
			for (int l = 0; l < numClass; l++) {
				if (train_target_temp.get(l) == 1) {
					LabelTemp.add(l);
					Label.set(i, LabelTemp);
				} else {
					not_LabelTemp.add(l);
					not_Label.set(i, not_LabelTemp);
				}
			}

		}
		System.out.println("OK Chunk2.");
		double[][] SVsFinal = transposeMatrix(SVs);
		this.support_vectors = SVsFinal;
		this.targetValues = train_target;
	}

	// Chunk3
	/*
	 */
	private void setKernelOptions(String str, double cost, double gamma,
			double degree) {

		this.cost = cost;
		if (str.equals("RBF")) {
			this.gamma = gamma;
			this.type = "RBF";
		} else if (str.equals("Poly")) {
			this.degree = degree;
			this.gamma = gamma;
			this.type = "Polynomial";
		} else {
			this.type = "Linear";
		}
	}

	private void KernelsSetup(MultiLabelInstances trainingSet, RealMatrix SVs) {

		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();
		double[][] kernel = new double[numTraining][numTraining];
		// Initialize kernel with 0s.
		for (double[] row : kernel)
			Arrays.fill(row, (double) 0);
		BlockRealMatrix SVs_copy = this.SVs;

		if (this.type.equals("RBF")) {
			for (int i = 0; i < numTraining; i++) {
				RealVector colVectorTemp1 = SVs_copy.getColumnVector(i);
				for (int j = 0; j < numTraining; j++) {
					RealVector colVectorTemp2 = SVs_copy.getColumnVector(j);
					RealVector SubtractionTemp = colVectorTemp1.subtract(colVectorTemp2);
					RealVector PowTemp = SubtractionTemp
							.mapToSelf(new Power(2));
					double sumTemp = StatUtils.sum(PowTemp.toArray());
					//for (int k = 0; k < numClass; k++) {
					//	sumTemp = sumTemp + PowTemp.getEntry(k);
					//}
					double MultTemp = (-this.gamma)*sumTemp;
					double ExpTemp = FastMath.exp(MultTemp);
					kernel[i][j] = ExpTemp;
				}
			}
		}
		RealMatrix RBFKernel =MatrixUtils.createRealMatrix(kernel);
		System.out.println("OK RBF.");
		
	}



	public static double[][] transposeMatrix(double[][] m) {
		double[][] temp = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	private void trainingChunk1(int sizeAlphaSum){
		//%Begin training phase

		this.alpha = new ArrayRealVector(sizeAlphaSum, 0);

		this.cValue = new ArrayList<BlockRealMatrix>();
		//cValue.
		int numClass = target.getColumnDimension();
		for (int i = 0; i < numClass; i++){
			BlockRealMatrix newAlpha = new BlockRealMatrix(numClass, numClass);
			ArrayRealVector rowVector = new ArrayRealVector(sizeAlphaSum, 1);
			newAlpha.setRowVector(i, rowVector);
			ArrayRealVector columnVector = new ArrayRealVector(sizeAlphaSum, -1);
			newAlpha.setColumnVector(i, columnVector);
			this.cValue.add(newAlpha);
		}
		System.out.println("OK training chunk 1.");
	}
	
	private void findAlpha(ArrayRealVector sizeAlpha, ArrayRealVector labelSize, ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> notLabel){
		boolean continuing = true;
		int iteration = 0;
	    System.out.println("current iteration: " + iteration);
		while (continuing){
		    iteration++;
			computeBeta(sizeAlpha, labelSize, Label, notLabel);
		//	computeGradient();
		//	findAlphaNew();
		//	findLambda();
		//	continuing = testConvergence();
		}
	}
	
	private void computeBeta(ArrayRealVector sizeAlpha, ArrayRealVector labelSize, ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> notLabel){
	    //Beta=zeros(num_class,num_training);
		int numClass = this.trainingSet.getNumLabels();
		int numTraining = this.trainingSet.getNumInstances();
		BlockRealMatrix beta = new BlockRealMatrix(numClass, numTraining);
		for (int k = 0; k < numClass; k++){
			for (int i = 0; i < numTraining; i++){
				double sum = StatUtils.sum(sizeAlpha.getSubVector(0, i-1).toArray());
				for (int m = 0; m < labelSize.getEntry(i); m++){
					for (int n = 0; n < numClass - labelSize.getEntry(i); n++){
						int index = new Double(sum + (m - 1) * (numClass - labelSize.getEntry(i)) + n).intValue();
						double oldBetaVal = beta.getEntry(k, i);
						double alphaVal = alpha.getEntry(index);
						int labelIndex = new Double(Label.get(i).getEntry(m)).intValue();
						int notLabelIndex = new Double(notLabel.get(i).getEntry(n)).intValue();
						double cv = cValue.get(k).getEntry(labelIndex, notLabelIndex);
						double newBetaVal = oldBetaVal + cv * alphaVal;
						beta.setEntry(k, i, newBetaVal);
					}
				}
			}
		}
	}

	private void computeBias(){
		//stub
	}
	private void computeSizePredictor(){
		//stub
	}
	
	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		return null;
	}

	public String globalInfo() {
		return "Class implementing the RankSVM algorithm." + "\n\n"
				+ "For more information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Elisseeff, Andr{\'e} and Weston, Jason");
		result.setValue(Field.TITLE,
				"A kernel method for multi-labelled classification");
		result.setValue(Field.BOOKTITLE,
				"Advances in neural information processing systems");
		result.setValue(Field.PAGES, "681--687");
		result.setValue(Field.YEAR, "2001");

		return result;
	}

}
