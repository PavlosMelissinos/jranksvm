package mlpms;

import java.util.ArrayList;
import java.util.Random;
import java.util.stream.IntStream;

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
	
    /** Train labels. */
    //protected double[] m_class;
	
   
    
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {

	}

	
	private void chuck1Lilia(){
		int[] labelIndices = trainingSet.getLabelIndices();
		//dataset preprocessing
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();
		int numAttributes = trainingSet.getFeatureIndices().length;
		double[][] SVs = new double[numAttributes][numTraining];
		int[] train_target = new int[numClass];
		int [][]indicesToKeep = new int[numAttributes][numTraining];
				
		for (Instance inst : trainingSet.getDataSet()){
			int j=1;
			int[] labels = new int[labelIndices.length];
			for (int i = 0; i < labelIndices.length; i++){
				labels[i] = (int) ( inst.value(labelIndices[i]));
				int sumTemp = IntStream.of((int) labels[i]).sum();
				if ((sumTemp!=numClass) && (sumTemp!=-numClass))
				{
					//double [] tempInst = inst.toDoubleArray(); 	
				    //SVs[i][j]=inst.getValueAt(i,j);
					//System.out.println("The sum is " + -numClass);
					//System.out.println("The sum is " + SVs.length);
					//System.out.println("The sum is " + SVs[0].length);
				    train_target[i]=labels[i];
				    j = j+1;
					//System.out.println("The sum is " + sumTemp);
				}
			}
		
	}}
	

	private int length(int[] featureIndices) {
		// TODO Auto-generated method stub
		return 0;
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
