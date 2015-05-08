package mlpms;

import java.awt.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math.linear.MatrixUtils;
import org.apache.commons.math.linear.RealMatrix;

import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.classifiers.functions.supportVector.SMOset;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

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

	private RealMatrix SVs;

	private RealMatrix target;

	/** Train labels. */
	// protected double[] m_class;

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		PreprocessingStep1(trainingSet);
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

		double[][] SVsArray = new double[numTraining][numFeatures];
		double[][] targetArray = new double[numTraining][numClass];
		int omitted = 0;
		for (int i = 0; i < numTraining; i++) {
			Instance inst = insts.get(i);
			double sumTemp = 0;
			double[] SVTemp = new double[numFeatures];
			double[] targetTemp = new double[numClass];
			int labelsChecked = 0;
			for (int attr = 0; attr < numAttr; attr++) {
				if (labelIndices[labelsChecked] == attr) {
					targetTemp[labelsChecked] = inst.value(attr);
					sumTemp += inst.value(attr);
					labelsChecked++;
				} else
					SVTemp[attr - labelsChecked] = inst.value(attr);
			}
			// filter out every instance that contains all or none of the labels
			// (uninformative)
			if ((sumTemp != numClass) && (sumTemp != -numClass)) {
				SVsArray[i - omitted] = SVTemp;
				targetArray[i - omitted] = targetTemp;
			} else
				omitted++;
		}
		this.SVs = MatrixUtils.createRealMatrix(SVsArray);
		this.target = MatrixUtils.createRealMatrix(targetArray);
	}

	private void PreprocessingStep1(MultiLabelInstances trainingSet) {

		int[] labelIndices = trainingSet.getLabelIndices();
		// dataset preprocessing
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();
		int numAttributes = trainingSet.getFeatureIndices().length;
		double[][] SVs = new double[numTraining][numAttributes];
		int[][] train_target = new int[numTraining][numClass];

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
				System.arraycopy(inst.toDoubleArray(), 0, tempMat, 0, numAttributes);
				for (int k = 0; k < numAttributes; k++) {
					SVs[obs][k] = tempMat[k];
				}
				for (int l = 0; l < numLabels; l++) {
					train_target[obs][l] = labels[l];}
				obs++;
			} else {ommited++;}
			count++;
		}
		double[][] SVsNew = new double[numTraining - ommited][numAttributes];
		int[][] train_targetNew = new int[numTraining - ommited][numClass];
		for (int i = 0; i < numTraining - ommited; i++) {
			System.arraycopy(SVs[i], 0, SVsNew[i], 0, numAttributes);
			System.arraycopy(train_target[i], 0, train_targetNew[i], 0,
					numClass);
		}
		if (Arrays.deepEquals(SVs, SVsNew))
			System.out.println("Equal.");
		else
			System.out.println("Different.");
		if (Arrays.deepEquals(train_target, train_targetNew))
			System.out.println("Equal.");
		else
			System.out.println("Different.");
		System.out.println("OK");
	}

	private int length(int[] featureIndices) {
		return 0;
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
