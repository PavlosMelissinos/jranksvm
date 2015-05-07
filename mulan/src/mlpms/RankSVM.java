package mlpms;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.functions.SMO;
<<<<<<< HEAD
import weka.core.Attribute;
=======
import weka.classifiers.functions.supportVector.SMOset;
>>>>>>> branch 'master' of ssh://git@bitbucket.org/PavlosM/jranksvm.git
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
/* Lilia
		 for (int j = 0; j < trainingSet.getNumInstances(); j++) {
			 Instance inst = trainingSet.getNextInstance();
			 double train_labels= inst.value(labelIndices[j]);
     }       
		SMO classifier = new SMO();
		//classifier.buildClassifier(trainingSet.getDataSet());
*/
		
		int[] labelIndices = trainingSet.getLabelIndices();
		//dataset preprocessing
		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();
		SMOset SVs = new SMOset(numTraining);
		
		for (Instance inst : trainingSet.getDataSet()){
			double[] labels = new double[labelIndices.length];
			for (int i = 0; i < labelIndices.length; i++){
				labels[i] = inst.value(labelIndices[i]);
			}
		}
		
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
		
		//initialize lagrange multipliers (alpha)
		
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
