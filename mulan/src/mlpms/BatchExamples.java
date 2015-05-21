package mlpms;

//package mulan.examples;


import java.util.logging.Level;
import java.util.logging.Logger;

import com.joptimizer.functions.LinearMultivariateRealFunction;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
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
			MultiLabelInstances trainingSet;
			trainingSet = new MultiLabelInstances("data/yeast-train.arff", "data/yeast.xml");
            //int [] temp = trainingSet.getLabelIndices();
			//for (int j = 0; j < trainingSet.getNumInstances(); j++) {
   			//	Instance inst = trainingSet.getNextInstance();
			LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { -1., -1. }, 4);   			 
			RankSVM classifier = new RankSVM();
			classifier.build(trainingSet);


	    } catch (Exception ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	    }
    	runLinear();
    }
    static void runLinear(){
    	LinearProgram lp = new LinearProgram(new double[]{5.0,10.0});
    	lp.addConstraint(new LinearBiggerThanEqualsConstraint(new double[]{3.0,1.0}, 8.0, "c1"));
    	lp.addConstraint(new LinearBiggerThanEqualsConstraint(new double[]{0.0,4.0}, 4.0, "c2"));
    	lp.addConstraint(new LinearSmallerThanEqualsConstraint(new double[]{2.0,0.0}, 2.0, "c3"));
    	lp.setMinProblem(true);
    	LinearProgramSolver solver  = SolverFactory.newDefault();
    	double[] sol = solver.solve(lp);
    }
    
    static void legacyCode(){
		// Constraints x=<UB, x>=LB
//		ArrayRealVector coeffTemp = new ArrayRealVector(
//				60000);
//		Collection<LinearConstraint> constraints = new ArrayList<LinearConstraint>();
//		for (int k = 0; k < 60000; k++) {
//			coeffTemp.setEntry(k, 0);
//		}
//		for (int j = 0; j < 60000; j++) {
//			ArrayRealVector coeff = coeffTemp.copy();
//			coeff.setEntry(j, 1);
//
//			constraints.add(new LinearConstraint(coeff, Relationship.LEQ,
//					Math.random()));
//			constraints.add(new LinearConstraint(coeff, Relationship.GEQ, 0));
//			//coeff.setEntry(j, 0);
//			System.out.println("Iteration: " + j);
//		}
//
//		System.out.println("Iterations done.");
   		//Classifier base = new J48();
   		//}
   			 
   			// double train_labels= inst.value(trainingSet.(j));
     //   }       
//			test = new MultiLabelInstances("data\\testData\\yeast-test.arff",
//			 "data\\testData\\yeast.xml");
            
            //@SuppressWarnings("unused")
			//Instances train_data = train.getDataSet();
            ///@SuppressWarnings("unused")
			//Classifier base = new J48();
//			Classifier base = new J48();
//			BinaryRelevance br = new BinaryRelevance(base);
//			br.build(train);
//			
//			Evaluator eval = new Evaluator();
//			Evaluation results = eval.evaluate(br, test, train);
//			System.out.println(results);
/* Lilia

            LinearProgram lp = new LinearProgram(new double[]{5.0,10.0});
        	lp.addConstraint(new LinearBiggerThanEqualsConstraint(new double[]{3.0,1.0}, 8.0, "c1"));
        	lp.addConstraint(new LinearBiggerThanEqualsConstraint(new double[]{0.0,4.0}, 4.0, "c2"));
        	lp.addConstraint(new LinearSmallerThanEqualsConstraint(new double[]{2.0,0.0}, 2.0, "c3"));
        	lp.setMinProblem(true);
        	LinearProgramSolver solver  = SolverFactory.newDefault();
        	double[] sol = solver.solve(lp);
        }
        catch (InvalidDataFormatException ex) {
       Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
	   } catch (Exception ex) {
	        Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
       }
*/    	
    }
    
}
