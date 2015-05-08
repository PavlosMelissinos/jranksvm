package mlpms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

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
 <!-- globalinfo-start -->
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 <!-- technical-bibtex-end -->
 * @author pavlos
 *
 */

@SuppressWarnings("serial")
public class RankSVM extends MultiLabelLearnerBase{

    /** Train_data. */
    private MultiLabelInstances trainingSet;
	
    private RealMatrix SVs;
    private RealMatrix target;
    /** Train labels. */
    //protected double[] m_class;
	
   
    
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		/* Lilia
		for (int j = 0; j < trainingSet.getNumInstances(); j++) {
			Instance inst = trainingSet.getNextInstance();
			double train_labels= inst.value(labelIndices[j]);
		}       
		SMO classifier = new SMO();
		//classifier.buildClassifier(trainingSet.getDataSet());
		*/
		
		/*
		 * SVs=[];
		 * target=[];
		 * //[num_training,~]=size(train_data);
		 * //[num_class,~]=size(train_target);
		 * for i=1:num_training
		 * //    temp=train_target(:,i);
		 *     if((sum(temp)~=num_class)&&(sum(temp)~=-num_class))
		 *         SVs=[SVs,train_data(i,:)'];
		 *         target=[target,temp];
		 *     end
		 * end
		*/
		
		setup(trainingSet);
		
		//initialize lagrange multipliers (alpha)

	}
	
	void setup(MultiLabelInstances trainingSet){
		//Preprocessing - Initialize support vectors & targets (labels)
		
		int[] labelIndices = trainingSet.getLabelIndices();
		//ArrayList labelIndices = new ArrayList(Arrays.asList(tempLabels));
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();

		Instances insts = trainingSet.getDataSet();
		int numAttr = insts.numAttributes();
		int numFeatures = numAttr - numClass;
		
		double[][] SVsArray = new double[numTraining][numFeatures];
		double[][] targetArray = new double[numTraining][numClass];
		int omitted = 0;
		for (int i = 0; i < numTraining; i++){
			Instance inst = insts.get(i);
			double sumTemp = 0;
			double[] SVTemp = new double[numFeatures];
			double[] targetTemp = new double[numClass];
			int labelsChecked = 0;
			for (int attr = 0; attr < numAttr; attr++){
				if (labelIndices[labelsChecked] == attr){
					targetTemp[labelsChecked] = inst.value(attr);
					sumTemp += inst.value(attr);
					labelsChecked++;
				}
				else
					SVTemp[attr - labelsChecked] = inst.value(attr);
			}
			//filter out every instance that contains all or none of the labels (uninformative)
			if ((sumTemp != numClass) && (sumTemp != -numClass)){
				SVsArray[i - omitted] = SVTemp;
				targetArray[i - omitted] = targetTemp;
			}
			else omitted++;
		}
		this.SVs = MatrixUtils.createRealMatrix(SVsArray);
		this.target = MatrixUtils.createRealMatrix(targetArray);
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		return null;
	}
	
    public String globalInfo() {
        return "Class implementing the RankSVM algorithm." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
    }
    
	@Override
	public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        		  
        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Elisseeff, Andr{\'e} and Weston, Jason");
        result.setValue(Field.TITLE, "A kernel method for multi-labelled classification");
        result.setValue(Field.BOOKTITLE, "Advances in neural information processing systems");
        result.setValue(Field.PAGES, "681--687");
        result.setValue(Field.YEAR, "2001");

        return result;
	}

}