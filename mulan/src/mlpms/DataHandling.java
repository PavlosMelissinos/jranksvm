package mlpms;

import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.CrossValidationExperiment;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DataHandling {

	public static void main(String[] args) {
    /**
     * Executes this example
     *
     * @param args command-line arguments -arff and -xml
     */
		try {
			MultiLabelInstances train, test;
			train = new MultiLabelInstances("data/yeast-train.arff",
			 "data/yeast.xml");
			test = new MultiLabelInstances("data/yeast-test.arff",
			 "data/yeast.xml");
			
			
			Instances train_data = train.getDataSet();
			LabelsMetaData train_labels =  train.getLabelsMetaData();
			int [] indices = train.getLabelIndices();
			int[][] matrix = {
					{ 1, 2, 3 },
					{ 4, 5, 6 },
					{ 7, 8, 9 }
					};

			for (int i = 0; i < matrix.length; i++) {
			    for (int j = 0; j < matrix[0].length; j++) {
			    	System.out.print(matrix[i][j] + " ");
			    }
			    System.out.print("\n");
			}
					
			for (int i = 0; i < indices.length; i++) {
				//for (int j = 0; j < indices[0].length; j++) {
			        System.out.print(indices[i] + " ");
			    //}
			    //System.out.print("\n");
			}
			//train_data = train.getDataSet(train);
			System.out.println(train.getNumInstances());
			System.out.println(train.getNumLabels());
			System.out.println(train_data);
			//System.out.println(train_labels);
			System.out.println(indices);
			//Classifier base = new J48();
			//BinaryRelevance br = new BinaryRelevance(base);
			//br.build(train);
			
			//Evaluator eval = new Evaluator();
			//Evaluation results = eval.evaluate(br, test, train);
			//System.out.println(train.getNumLabels());
			//System.out.println(train.getDataSet());
	    } catch (InvalidDataFormatException ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	    } catch (Exception ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	    }	
    }
}