package mlpms;

//package mulan.examples;


import java.util.logging.Level;
import java.util.logging.Logger;

import mlpms.RankSVM.KernelType;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.CrossValidationExperiment;


public class BatchExamples {
    /**
     * Executes this example
     *
     * @param args command-line arguments -arff and -xml
     * @throws InvalidDataFormatException 
     */
    public static void main(String[] args) throws InvalidDataFormatException {
    	try {
			MultiLabelInstances trainingSet =
					new MultiLabelInstances("data/yeast-train-10percent.arff", "data/yeast.xml");
			RankSVM classifier = new RankSVM();
			classifier.setKernelOptions(KernelType.RBF, 1, 1, 1, 1);
			classifier.build(trainingSet);
			
			MultiLabelInstances testingSet =
					new MultiLabelInstances("data/yeast-test-10percent.arff", "data/yeast.xml");
            Evaluator eval = new Evaluator();
            Evaluation results = eval.evaluate(classifier, testingSet, trainingSet);
            System.out.println(results);


	    } catch (Exception ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	    }
    }
    
}
