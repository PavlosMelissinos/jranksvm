package mlpms;

//package mulan.examples;


import java.util.logging.Level;
import java.util.logging.Logger;

//import jeans.math.evaluate.Evaluator;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.CrossValidationExperiment;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;


public class BatchExamples {
    /**
     * Executes this example
     *
     * @param args command-line arguments -arff and -xml
     */
    public static void main(String[] args) {

    	System.out.println("Hello World");
    	System.out.println("Hello World2");
    	System.out.println("Hello World3");
    	System.out.println("Hello World4");
    	 try {
	MultiLabelInstances train, test;
	train = new MultiLabelInstances("data\\yeast-train.arff",
	 "data\\yeast.xml");
	test = new MultiLabelInstances("data\\testData\\yeast-test.arff",
	 "data\\testData\\yeast.xml");
	Classifier base = new J48();
	BinaryRelevance br = new BinaryRelevance(base);
	br.build(train);
	
	Evaluator eval = new Evaluator();
	Evaluation results = eval.evaluate(br, test, train);
	System.out.println(results);
    } catch (InvalidDataFormatException ex) {
        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
    } catch (Exception ex) {
        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
    }
 }
}
