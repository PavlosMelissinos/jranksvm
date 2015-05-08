package mlpms;

//package mulan.examples;


import java.util.logging.Level;
import java.util.logging.Logger;












import scpsolver.constraints.LinearBiggerThanEqualsConstraint;
import scpsolver.constraints.LinearSmallerThanEqualsConstraint;
import scpsolver.lpsolver.LinearProgramSolver;
import scpsolver.lpsolver.SolverFactory;
import scpsolver.problems.LinearProgram;
//import jeans.math.evaluate.Evaluator;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.examples.CrossValidationExperiment;
import weka.classifiers.Classifier;
import weka.classifiers.functions.supportVector.SMOset;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;


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
          //  for (int j = 0; j < trainingSet.getNumInstances(); j++) {
   			// Instance inst = trainingSet.getNextInstance();
   			 
   		RankSVM classifier = new RankSVM();
   		classifier.build(trainingSet);
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
    } catch (InvalidDataFormatException ex) {
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
    
}
