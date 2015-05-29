package mlpms;

//package mulan.examples;


import java.util.logging.Level;
import java.util.logging.Logger;

import com.joptimizer.functions.LinearMultivariateRealFunction;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.CrossValidationExperiment;
import scpsolver.constraints.LinearBiggerThanEqualsConstraint;
import scpsolver.constraints.LinearSmallerThanEqualsConstraint;
import scpsolver.lpsolver.LinearProgramSolver;
import scpsolver.lpsolver.SolverFactory;
import scpsolver.problems.LinearProgram;
//import jeans.math.evaluate.Evaluator;


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
					new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");
			RankSVM classifier = new RankSVM();
			classifier.build(trainingSet);
			classifier.build(trainingSet);
			
			MultiLabelInstances testingSet =
					new MultiLabelInstances("data/yeast-test.arff", "data/yeast.xml");
            Evaluator eval = new Evaluator();
            Evaluation results = eval.evaluate(classifier, testingSet, trainingSet);
            System.out.println(results);


	    } catch (Exception ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	    }
    }
    
}
