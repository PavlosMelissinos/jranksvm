package mlpms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.stream.DoubleStream;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.SearchInterval;
import org.apache.commons.math3.optim.univariate.UnivariateObjectiveFunction;
import org.apache.commons.math3.optim.univariate.UnivariatePointValuePair;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

import scpsolver.constraints.LinearBiggerThanEqualsConstraint;
import scpsolver.constraints.LinearEqualsConstraint;
import scpsolver.constraints.LinearSmallerThanEqualsConstraint;
import scpsolver.lpsolver.LinearProgramSolver;
import scpsolver.lpsolver.SolverFactory;
import scpsolver.problems.LinearProgram;
import scpsolver.util.SparseVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

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
	// private MultiLabelInstances trainingSet;

	/** Train_data (only features). */
	// private double[][] train_data;
	private BlockRealMatrix trainData;

	/** Train labels. */
	// private double[][] train_target;
	private BlockRealMatrix trainTarget;

	private BlockRealMatrix SVs;

	private BlockRealMatrix target;

	// private ArrayList<BlockRealMatrix> cValue;

	private ArrayRealVector alpha;

	private double cost;

	private double gamma;

	private double degree;
	private double lambdaTol = 1e-6;
	private double normTol = 1e-4;
	private int maxIter = 50;

	enum KernelType {
		LINEAR, POLYNOMIAL, RBF;
	}

	private KernelType kType;

	private double coefficient;

	/** Train labels. */

	public RankSVM() {
		this.cost = 1;
		this.gamma = 1;
		this.degree = 1;
		this.coefficient = 1;
		this.kType = KernelType.POLYNOMIAL;
	}

	public BlockRealMatrix getSVs() {
		return this.SVs;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		// PreprocessingStep1(trainingSet);
		HashMap<String, Object> alphas = setup(trainingSet);
		// setKernelOptions("RBF", 1, 1, 0, 0);
		// setKernelOptions("Polynomial", 1, 1, 1, 2);
		BlockRealMatrix kernel = KernelsSetup(trainingSet, getSVs());

		ArrayRealVector sizeAlpha = (ArrayRealVector) alphas.get("sizeAlpha");
		ArrayRealVector labelSize = (ArrayRealVector) alphas.get("labelSize");
		ArrayList<ArrayRealVector> Label = (ArrayList<ArrayRealVector>) alphas
				.get("Label");
		ArrayList<ArrayRealVector> notLabel = (ArrayList<ArrayRealVector>) alphas
				.get("notLabel");
		ArrayRealVector gradient = findAlpha(sizeAlpha, labelSize, Label, notLabel, kernel);
		ArrayRealVector bias = computeBias(labelSize, sizeAlpha, Label, notLabel, gradient);
		// computeSizePredictor();
	}

	HashMap<String, Object> setup(MultiLabelInstances trainingSet) {
		// Preprocessing - Initialize support vectors & targets (labels)

		int[] labelIndices = trainingSet.getLabelIndices();

		int numTraining = trainingSet.getNumInstances();
		int numClass = trainingSet.getNumLabels();

		Instances insts = trainingSet.getDataSet();
		int numAttr = insts.numAttributes();
		int numFeatures = numAttr - numClass;

		SVs = new BlockRealMatrix(numTraining, numFeatures);
		target = new BlockRealMatrix(numClass, numTraining);
		this.trainData = new BlockRealMatrix(numTraining, numFeatures);
		this.trainTarget = new BlockRealMatrix(numClass, numTraining);
		int omitted = 0;
		for (int i = 0; i < numTraining; i++) {
			Instance inst = insts.get(i);
			double sumTemp = 0;
			ArrayRealVector SVRow = new ArrayRealVector(numFeatures);
			ArrayRealVector targetColumn = new ArrayRealVector(numClass);
			int labelsChecked = 0;
			for (int attr = 0; attr < numAttr; attr++) {
				if (labelIndices[labelsChecked] == attr) {
					targetColumn.setEntry(labelsChecked, inst.value(attr));
					sumTemp += inst.value(attr);
					labelsChecked++;
				} else
					SVRow.setEntry(attr - labelsChecked, inst.value(attr));
			}
			// filter out every instance that contains all or none of the labels
			// (uninformative)
			this.trainData.setRowVector(i, SVRow);
			this.trainTarget.setColumnVector(i, targetColumn);
			if ((sumTemp != numClass) && (sumTemp != -numClass)) {
				SVs.setRowVector(i - omitted, SVRow);
				target.setColumnVector(i - omitted, targetColumn);
			} else
				omitted++;
		}
		this.SVs.getSubMatrix(0, numTraining - omitted - 1, 0, numFeatures - 1);
		this.SVs = this.SVs.transpose(); // numInstances x numFeatures -->
											// numFeatures x numInstances
		this.target.getSubMatrix(0, numClass - 1, 0, numTraining - omitted - 1);

		// Chunk2

		ArrayRealVector labelSize = new ArrayRealVector(numTraining);
		ArrayRealVector sizeAlpha = new ArrayRealVector(numTraining);

		ArrayList<ArrayRealVector> Label = new ArrayList<ArrayRealVector>();
		ArrayList<ArrayRealVector> notLabel = new ArrayList<ArrayRealVector>();
		/* Initialize ArrayList of ArrayLists to have specific number of rows. */

		for (int i = 0; i < numTraining - omitted; i++) {
			ArrayRealVector trainLabelRow = (ArrayRealVector) trainTarget
					.getColumnVector(i); // temp label vector
			double labelSum = DoubleStream.of(trainLabelRow.toArray()).sum();
			labelSize.setEntry(i, labelSum);
			sizeAlpha.setEntry(i, labelSum * (numClass - labelSum));

			ArrayList<Double> LabelTemp = new ArrayList<Double>();
			ArrayList<Double> notLabelTemp = new ArrayList<Double>();
			for (int l = 0; l < numClass; l++) {
				if (trainLabelRow.getEntry(l) == 1)
					LabelTemp.add((double) l);
				else
					notLabelTemp.add((double) l);
			}

			Double[] labelArr = new Double[LabelTemp.size()];
			labelArr = LabelTemp.toArray(labelArr);
			Label.add(new ArrayRealVector(labelArr));

			Double[] notLabelArr = new Double[notLabelTemp.size()];
			notLabelArr = notLabelTemp.toArray(notLabelArr);
			notLabel.add(new ArrayRealVector(notLabelArr));
		}

		HashMap<String, Object> results = new HashMap<String, Object>();
		results.put("sizeAlpha", sizeAlpha);
		results.put("labelSize", labelSize);
		results.put("Label", Label);
		results.put("notLabel", notLabel);

		return results;
	}

	// Chunk3
	void setKernelOptions(KernelType kType, double cost, double gamma,
			double coefficient, double degree) {

		this.cost = cost;
		this.kType = kType;
		switch (kType) {
		case POLYNOMIAL:
			this.degree = degree;
			this.gamma = gamma;
			this.coefficient = coefficient;
			break;
		case RBF:
			this.gamma = gamma;
			break;
		default:
			break;
		}
	}

	private BlockRealMatrix KernelsSetup(MultiLabelInstances trainingSet,
			RealMatrix SVs) {

		int numTraining = trainingSet.getNumInstances();
		BlockRealMatrix kernel = new BlockRealMatrix(numTraining, numTraining);
		BlockRealMatrix SVs_copy = this.SVs;

		if (this.kType == KernelType.RBF) {
			for (int i = 0; i < numTraining; i++) {
				RealVector colVectorTemp1 = SVs_copy.getColumnVector(i);
				for (int j = 0; j < numTraining; j++) {
					RealVector colVectorTemp2 = SVs_copy.getColumnVector(j);
					RealVector SubtractionTemp = colVectorTemp1
							.subtract(colVectorTemp2);
					RealVector PowTemp = SubtractionTemp
							.mapToSelf(new Power(2));
					double sumTemp = StatUtils.sum(PowTemp.toArray());
					double MultTemp = (-this.gamma) * sumTemp;
					double ExpTemp = FastMath.exp(MultTemp);
					kernel.setEntry(i, j, ExpTemp);
				}
			}
			System.out.println("OK RBF.");
		} else if (this.kType == KernelType.POLYNOMIAL) {
			for (int i = 0; i < numTraining; i++) {
				RealVector colVectorTemp1 = SVs_copy.getColumnVector(i);
				for (int j = 0; j < numTraining; j++) {
					RealVector colVectorTemp2 = SVs_copy.getColumnVector(j);
					double MultTemp1 = colVectorTemp1
							.dotProduct(colVectorTemp2);
					double MultTemp2 = MultTemp1 * (this.gamma);
					double AddTemp = MultTemp2 + this.coefficient;
					double PowTemp = Math.pow(AddTemp, this.degree);
					kernel.setEntry(i, j, PowTemp);
				}
			}
			System.out.println("OK Polynomial.");
		} else if (this.kType == KernelType.LINEAR) {
			for (int i = 0; i < numTraining; i++) {
				RealVector colVectorTemp1 = SVs_copy.getColumnVector(i);
				for (int j = 0; j < numTraining; j++) {
					RealVector colVectorTemp2 = SVs_copy.getColumnVector(j);
					double MultTemp1 = colVectorTemp1
							.dotProduct(colVectorTemp2);
					kernel.setEntry(i, j, MultTemp1);
				}
			}
			System.out.println("OK Linear.");
		}
		return kernel;
	}

	private ArrayList<BlockRealMatrix> trainingChunk1() {
		// %Begin training phase
		ArrayList<BlockRealMatrix> cValue = new ArrayList<BlockRealMatrix>();

		int numClass = target.getRowDimension();
		for (int i = 0; i < numClass; i++) {
			BlockRealMatrix newAlpha = new BlockRealMatrix(numClass, numClass);
			ArrayRealVector rowVector = new ArrayRealVector(numClass, 1);
			newAlpha.setRowVector(i, rowVector);
			ArrayRealVector columnVector = new ArrayRealVector(numClass, -1);
			newAlpha.setColumnVector(i, columnVector);
			cValue.add(newAlpha);
		}
		System.out.println("OK training chunk 1.");
		return cValue;
	}

	private ArrayRealVector findAlpha(ArrayRealVector sizeAlpha,
			ArrayRealVector labelSize, ArrayList<ArrayRealVector> Label,
			ArrayList<ArrayRealVector> notLabel, BlockRealMatrix kernel) {
		boolean continuing = true;

		ArrayList<BlockRealMatrix> cValue = trainingChunk1();

		int sizeAlphaSum = (int) DoubleStream.of(sizeAlpha.toArray())
				.map(Math::round).sum();
		this.alpha = new ArrayRealVector(sizeAlphaSum, 0);
		int numClass = this.trainTarget.getRowDimension();
		int numTraining = this.trainTarget.getColumnDimension();
		double lambda = 0;
		ArrayRealVector gradient = new ArrayRealVector(sizeAlphaSum);
		for (int iteration = 1; continuing; iteration++) {
			System.out.println("current iteration: " + iteration);

			// computeBeta(sizeAlpha, labelSize, Label, notLabel);
			BlockRealMatrix beta = new BlockRealMatrix(numClass, numTraining);
			for (int k = 0; k < numClass; k++) {
				for (int i = 0; i < numTraining; i++) {
					double sum = i > 0 ? StatUtils.sum(sizeAlpha.getSubVector(
							0, i).toArray()) : 0;
					for (int m = 0; m < labelSize.getEntry(i); m++) {
						for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
							int index = new Double(sum + m
									* (numClass - labelSize.getEntry(i)) + n
									+ 1).intValue();
							// System.out.println(k + " + " + i + " + " + m +
							// " + " + n + " + " + index);
							double oldBetaVal = beta.getEntry(k, i);
							double alphaVal = alpha.getEntry(index - 1);
							int labelIndex = new Double(Label.get(i)
									.getEntry(m)).intValue();
							int notLabelIndex = new Double(notLabel.get(i)
									.getEntry(n)).intValue();
							double cv = cValue.get(k).getEntry(labelIndex,
									notLabelIndex);
							double newBetaVal = oldBetaVal + cv * alphaVal;
							beta.setEntry(k, i, newBetaVal);
						}
					}
				}
			}

		    
		//	computeGradient
		    
		    BlockRealMatrix inner = beta.multiply(kernel); //inner = beta X kernel
															// kernel
			gradient = new ArrayRealVector(sizeAlphaSum);
			int index = 0;
			for (int i = 0; i < numTraining; i++) {
				for (int m = 0; m < labelSize.getEntry(i); m++) {
					int labelIndex = new Double(Label.get(i).getEntry(m))
							.intValue();
					for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
						int notLabelIndex = new Double(notLabel.get(i)
								.getEntry(n)).intValue();
						double temp = inner.getEntry(labelIndex, i)
								- inner.getEntry(notLabelIndex, i) - 1;
						gradient.setEntry(index++, temp);
					}
				}
			}

			// compute Alpha_new
			ArrayRealVector Alpha_new = findAlphaNew(gradient, numClass,
					sizeAlphaSum, numTraining, labelSize, Label, notLabel,
					cValue, sizeAlpha);

			lambda = findLambda(Alpha_new, cValue, kernel, numTraining,
					numClass, Label, notLabel, labelSize, sizeAlpha);

			continuing = testConvergence(lambda, iteration, Alpha_new);
		}
		return gradient;
	}

	ArrayRealVector findAlphaNew(ArrayRealVector gradient, int numClass,
			int sizeAlphaSum, int numTraining, ArrayRealVector labelSize,
			ArrayList<ArrayRealVector> Label,
			ArrayList<ArrayRealVector> notLabel,
			ArrayList<BlockRealMatrix> cValue, ArrayRealVector sizeAlpha) {

		BlockRealMatrix Aeq = new BlockRealMatrix(numClass, sizeAlphaSum);

		for (int k = 0; k < numClass; k++) {
			int counter = 0;
			for (int i = 0; i < numTraining; i++) {
				for (int m = 0; m < labelSize.getEntry(i); m++) {
					for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
						double temp1 = Label.get(i).getEntry(m);
						double temp2 = notLabel.get(i).getEntry(n);
						double temp = cValue.get(k).getEntry((int) temp1,
								(int) temp2);
						Aeq.addToEntry(k, counter++, temp);
					}
				}
			}
		}
		ArrayRealVector beq = new ArrayRealVector(numClass);
		ArrayRealVector UB = new ArrayRealVector(sizeAlphaSum);
		int counter = 0;
		for (int i = 0; i < numTraining; i++) {
			for (int m = 0; m < labelSize.getEntry(i); m++) {
				for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
					double temp = cost / sizeAlpha.getEntry(i);
					UB.addToEntry(counter++, temp);
				}
			}
		}
		System.out.println("Lilia: ");
		// find alpha new
		LinearObjectiveFunction f = new LinearObjectiveFunction(gradient, 0);
		ArrayList<LinearConstraint> constraints = new ArrayList<LinearConstraint>();
		//Set<LinearConstraint> constraints = new LinkedHashSet<LinearConstraint>();
		
		
		// Constraint Aeq*x=beq
		for (int i = 0; i < Aeq.getRowDimension(); i++) {
			double[] Av = new double[Aeq.getColumnDimension()];
			for (int j = 0; j < Aeq.getColumnDimension(); j++) {
				Av[j] = Aeq.getEntry(i, j);
			}
			constraints.add(new LinearConstraint(Av, Relationship.EQ,
					beq.getEntry(i)));
		}

		// Constraints x=<UB, x>=LB
		ArrayRealVector coeffTemp = new ArrayRealVector(
				Aeq.getColumnDimension());
		for (int j = 0; j < Aeq.getColumnDimension(); j++) {
			//ArrayRealVector coeff = coeffTemp;//.copy();
			OpenMapRealVector coeff = new OpenMapRealVector(Aeq.getColumnDimension());
			coeff.setEntry(j, 1);
			constraints.add(new LinearConstraint(coeff, Relationship.LEQ, UB.getEntry(j)));
			constraints.add(new LinearConstraint(coeff, Relationship.GEQ, 0));
			//coeff.setEntry(j, 0);
			System.out.println("Iteration: " + j);
		}
		HashSet<LinearConstraint> consts = new HashSet<LinearConstraint>(3 * constraints.size() / 2);
		Collections.addAll(consts, (LinearConstraint[]) constraints.toArray());
		 //= new HashSet<LinearConstraint>(constraints);
		SimplexSolver solver = new SimplexSolver();
		PointValuePair optSolution = solver.optimize(new MaxIter(100), f,
				//new LinearConstraintSet(constraints), GoalType.MINIMIZE, new SimpleBounds(LB, UB.toArray()));
				new LinearConstraintSet(constraints), GoalType.MINIMIZE);
		// solution = optSolution.getPoint();
		ArrayRealVector solution = new ArrayRealVector(optSolution.getPoint());
		System.out.println("Ok: ");
		return solution;
		
//    	LinearProgram lp = new LinearProgram(gradient.toArray());
//		for (int i = 0; i < Aeq.getRowDimension(); i++) {
//	    	lp.addConstraint(new LinearEqualsConstraint(Aeq.getRow(i), beq.getEntry(i), "c1"));
//		}
//		for (int j = 0; j < Aeq.getColumnDimension(); j++) {
//			SparseVector coeff = new SparseVector(Aeq.getColumnDimension(), 1);
//			coeff.set(j, 1);
//	    	lp.addConstraint(new LinearBiggerThanEqualsConstraint(coeff, 0, "c2"));
//	    	lp.addConstraint(new LinearSmallerThanEqualsConstraint(coeff, UB.getEntry(j), "c3"));
//	    	System.out.println("Iteration " + j);
//		}
//    	lp.setMinProblem(true);
//    	LinearProgramSolver solver  = SolverFactory.newDefault();
//    	double[] sol = solver.solve(lp);
//    	return new ArrayRealVector(sol);
	}

	double findLambda(ArrayRealVector Alpha_new,
			ArrayList<BlockRealMatrix> cValue, BlockRealMatrix kernel,
			int numTraining, int numClass, ArrayList<ArrayRealVector> Label,
			ArrayList<ArrayRealVector> notLabel, ArrayRealVector labelSize,
			ArrayRealVector sizeAlpha) {
		NegDualFuncUniWrapper f = new NegDualFuncUniWrapper(alpha, Alpha_new,
				cValue, kernel, numTraining, numClass, Label, notLabel,
				labelSize, sizeAlpha);
		BrentOptimizer solver = new BrentOptimizer(1e-10, 1e-14);
		UnivariatePointValuePair solution = solver.optimize(new MaxEval(200),
				new UnivariateObjectiveFunction(f), GoalType.MINIMIZE,
				new SearchInterval(0, 1));
		double lambda = solution.getPoint();
		return lambda;
	}

	private boolean testConvergence(double lambda, int iteration,
			ArrayRealVector Alpha_new) {
		boolean continuing = true;

		double dist = Alpha_new.getDistance(alpha);
		if (Math.abs(lambda) <= lambdaTol || lambda * dist <= normTol) {
			continuing = false;
			System.out.println("program terminated normally");
		}
		else if (iteration >= maxIter) {
		    continuing = false;
			System.err.println("maximum number of iterations reached, procedure not convergent");
		}
		else
			alpha = alpha.add(Alpha_new.subtract(alpha).mapMultiplyToSelf(lambda));
		return continuing;
	}

	private ArrayRealVector computeBias(ArrayRealVector labelSize, ArrayRealVector sizeAlpha,
			ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> notLabel, ArrayRealVector gradient) {
		int numClass = this.trainTarget.getRowDimension();
		int numTraining = this.trainTarget.getColumnDimension();
		int sizeAlphaSum = new Double(DoubleStream.of(sizeAlpha.toArray()).sum()).intValue();
		BlockRealMatrix left = new BlockRealMatrix(sizeAlphaSum, numClass);
		ArrayRealVector right = new ArrayRealVector(sizeAlphaSum);
		int counter = 0;
		for (int i = 0; i < numTraining; i++){
		    for (int m = 0; m < labelSize.getEntry(i); m++){
		        for (int n = 0; n < numClass - labelSize.getEntry(i); n++){
//		            index = sum(size_alpha(1:i - 1)) + (m - 1) * (num_class - Label_size(i)) + n;
		        	int sizeAlphaSubSum = (int) DoubleStream.of(sizeAlpha.getSubVector(0, i - 1).toArray()).map(Math::round).sum();
		        	int index = sizeAlphaSubSum + (m - 1) * (numClass - new Double(labelSize.getEntry(i)).intValue()) + n;
//		            if((abs(Alpha(index))>=lambda_tol)&&(abs(Alpha(index)-(cost/(size_alpha(i))))>=lambda_tol))
		        	if (Math.abs(alpha.getEntry(index)) >= lambdaTol &&
		        			Math.abs(alpha.getEntry(index) - (cost / sizeAlpha.getEntry(i))) >= lambdaTol){
		        	
		        		ArrayRealVector vector = new ArrayRealVector(numClass);
			        	vector.setEntry(new Double(Label.get(i).getEntry(m)).intValue(), 1.0);
			        	vector.setEntry(new Double(notLabel.get(i).getEntry(m)).intValue(), -1.0);
			        	
			        	left.setRowVector(counter++, vector);
		                right.setEntry(counter++, -gradient.getEntry(index));
		        	}
		        }
		    }
		}
		ArrayRealVector bias;
		if (left.getColumnDimension() == 0 || left.getRowDimension() == 0){
			bias = new ArrayRealVector(trainTarget.getRowDimension());
			for (int i = 0; i < trainTarget.getRowDimension(); i++){
				int sum = new Double(DoubleStream.of(trainTarget.getRowVector(i).toArray()).sum()).intValue();
				bias.setEntry(i, sum);
			}
		}
		else{
			// Left * bias' = Right
			DecompositionSolver solver = new LUDecomposition(left).getSolver();
			bias = (ArrayRealVector) solver.solve(right);
		}
		return bias;
	}

	private void computeSizePredictor() {
		// stub
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
