package mlpms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.PointValuePair;
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
import org.apache.commons.math3.util.Precision;

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


	private BlockRealMatrix target;

	private BlockRealMatrix weights;
	private ArrayRealVector bias;
	private BlockRealMatrix SVs;
	private ArrayRealVector weightsSizePre;
	private double biasSizePre;

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
	
	public RankSVM(BlockRealMatrix weights, ArrayRealVector bias, 
			BlockRealMatrix SVs, ArrayRealVector weightsSizePre, double biasSizePre){
		this.weights = weights;
		this.bias = bias;
		this.SVs = SVs;
		this.weightsSizePre = weightsSizePre;
		this.biasSizePre = biasSizePre;
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
		HashMap<String, Object> alphas = setup(trainingSet);
		BlockRealMatrix kernel = KernelsSetup(trainingSet, getSVs());

		ArrayRealVector sizeAlpha = (ArrayRealVector) alphas.get("sizeAlpha");
		ArrayRealVector labelSize = (ArrayRealVector) alphas.get("labelSize");
		ArrayList<ArrayRealVector> Label = (ArrayList<ArrayRealVector>) alphas.get("Label");
		ArrayList<ArrayRealVector> notLabel = (ArrayList<ArrayRealVector>) alphas.get("notLabel");
		ArrayRealVector gradient = findAlpha(sizeAlpha, labelSize, Label, notLabel, kernel);
		computeBias(labelSize, sizeAlpha, Label, notLabel, gradient);
		computeSizePredictor(weights, bias, kernel, Label, notLabel);
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
	protected void setKernelOptions(KernelType kType, double cost, double gamma,
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

	BlockRealMatrix KernelsSetup(MultiLabelInstances trainingSet,
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

	ArrayList<BlockRealMatrix> trainingChunk1() {
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

	 ArrayRealVector findAlpha(ArrayRealVector sizeAlpha,
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
		BlockRealMatrix beta = null;
		for (int iteration = 1; continuing; iteration++) {
			System.out.println("current iteration: " + iteration);

			// computeBeta(sizeAlpha, labelSize, Label, notLabel);
			beta = new BlockRealMatrix(numClass, numTraining);
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
		this.weights = beta;
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
			OpenMapRealVector coeff = new OpenMapRealVector(Aeq.getColumnDimension());
			coeff.setEntry(j, 1);
			constraints.add(new LinearConstraint(coeff, Relationship.LEQ, UB.getEntry(j)));
			constraints.add(new LinearConstraint(coeff, Relationship.GEQ, 0));
		}
		HashSet<LinearConstraint> consts = new HashSet<LinearConstraint>(constraints);
		SimplexSolver solver = new SimplexSolver();
		PointValuePair optSolution = solver.optimize(new MaxIter(100), f,
				new LinearConstraintSet(consts), GoalType.MINIMIZE);
		ArrayRealVector solution = new ArrayRealVector(optSolution.getPoint());
		System.out.println("Ok: ");
		return solution;
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

	 boolean testConvergence(double lambda, int iteration,
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

	 void computeBias(ArrayRealVector labelSize, ArrayRealVector sizeAlpha,
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
		        	int sizeAlphaSubSum = (int) DoubleStream.of(sizeAlpha.getSubVector(0, i - 1).toArray()).map(Math::round).sum();
		        	int index = sizeAlphaSubSum + (m - 1) * (numClass - new Double(labelSize.getEntry(i)).intValue()) + n;
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
		if (left.getColumnDimension() == 0 || left.getRowDimension() == 0){
			this.bias = new ArrayRealVector(trainTarget.getRowDimension());
			for (int i = 0; i < trainTarget.getRowDimension(); i++){
				int sum = new Double(DoubleStream.of(trainTarget.getRowVector(i).toArray()).sum()).intValue();
				this.bias.setEntry(i, sum);
			}
		}
		else{
			// Left * bias' = Right
			DecompositionSolver solver = new LUDecomposition(left).getSolver();
			this.bias = (ArrayRealVector) solver.solve(right);
		}
	}

	void computeSizePredictor(BlockRealMatrix beta, ArrayRealVector bias, BlockRealMatrix kernel, ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> notLabel) {
//		Computing the size predictor using linear least squares model [2]

		int numClass = this.trainTarget.getRowDimension();
		int numTraining = this.trainTarget.getColumnDimension();
		BlockRealMatrix left = new BlockRealMatrix(numTraining, numClass);
				
		left = beta.multiply(kernel).add(bias.outerProduct(new ArrayRealVector(bias.getDimension(), 1.0))).transpose();

		ArrayRealVector right = new ArrayRealVector(numTraining);
		for (int i = 0; i < numTraining; i++){
			int counter = 0;
			ArrayList<ValueIndexPair> temp = new ArrayList<ValueIndexPair>();
			//TreeMap<Integer, Double> temp = new TreeMap<Integer, Double>();
			for (double entry: left.getRow(i)){
				temp.add(new ValueIndexPair(counter++, entry));
			}
			Collections.sort(temp);
			ArrayRealVector values = new ArrayRealVector(temp.size());
			ArrayRealVector indices = new ArrayRealVector(temp.size());
			counter = 0;
			for (ValueIndexPair entry: temp){
				values.setEntry(counter, entry.value);
				indices.setEntry(counter++, entry.index);
			}
			
			ArrayRealVector candidate = new ArrayRealVector(numClass + 1);
			candidate.setEntry(1, temp.get(0).value - 0.1);

			for (int j = 0; j < numClass - 1; j++){
				double val = (values.getEntry(j) + values.getEntry(j + 1)) / 2;
				candidate.setEntry(j + 1, val);
			}
			candidate.setEntry(numClass + 1, temp.get(numClass).value + 0.1);
			ArrayRealVector missClass = new ArrayRealVector(numClass + 1);
			for (int j = 0; j < numClass + 1; j++){
				ArrayRealVector tempNotLabels = new ArrayRealVector();
				if (j > 0){
					tempNotLabels = (ArrayRealVector) indices.getSubVector(0, j);
				}
				ArrayRealVector tempLabels = new ArrayRealVector();
				if (j < numClass)
					tempLabels = (ArrayRealVector) indices.getSubVector(j, numClass - j);
				
				int falseNeg = setDiff(tempNotLabels, notLabel.get(i)).size();
				int falsePos = setDiff(tempLabels, Label.get(i)).size();
				missClass.setEntry(j, falseNeg + falsePos);
				
			}
			
			int tempMinimum = new Double(missClass.getMinValue()).intValue();
			int tempIndex = new Double(missClass.getMinIndex()).intValue();
			
			right.setEntry(i, candidate.getEntry(tempIndex));
		}
		BlockRealMatrix leftNew = new BlockRealMatrix(numTraining, numClass + 1);
		for (int j = 0; j < numClass; j++){
			leftNew.setColumnVector(j, left.getColumnVector(j));
		}
		leftNew.setColumnVector(numClass, new ArrayRealVector(numTraining, 1));
		
		DecompositionSolver solver = new LUDecomposition(left).getSolver();
		ArrayRealVector tempValue = (ArrayRealVector) solver.solve(right);
		
		this.weightsSizePre = (ArrayRealVector) tempValue.getSubVector(0, numClass);
		
		this.biasSizePre = tempValue.getEntry(numClass);
		
	}
	
	Set<Double> setDiff(ArrayRealVector a, ArrayRealVector b){
		Set<Double> aa = DoubleStream.of(a.getDataRef())
				.boxed().collect(Collectors.toSet());
		Set<Double> bb = DoubleStream.of(b.getDataRef())
				.boxed().collect(Collectors.toSet());
		aa.removeAll(bb);
		return aa;
		
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		int numTraining = this.SVs.getColumnDimension();
		int numClass = this.numLabels;

	    double labelSize = DoubleStream.of(instance.toDoubleArray()).sum();
		ArrayRealVector label = new ArrayRealVector(Double.valueOf(labelSize).intValue());
		ArrayRealVector notLabel = new ArrayRealVector(Double.valueOf(labelSize).intValue());
	    ArrayRealVector instVector = new ArrayRealVector(instance.toDoubleArray());
	    int labelIndex = 0;
	    for (int j = 0; j < numClass; j++){
	        if (Precision.equals(instVector.getEntry(j), 1.0))
	            label.setEntry(labelIndex++, j);
	        else notLabel.setEntry(j - labelIndex, j);
	    }
	    
	    ArrayRealVector kernel = setupPredictKernel(instance);
	    
	    ArrayRealVector outputs = new ArrayRealVector(numClass);
        for (int k = 0; k < numClass; k++){
            double temp = 0;
            for (int j = 0; j < numTraining; j++){
                temp = temp + weights.getEntry(k,j) * kernel.getEntry(j);            
            }
            temp = temp + bias.getEntry(k);
            outputs.setEntry(k, temp);
        }
        double threshold = outputs.append(1).dotProduct(this.weightsSizePre.append(this.biasSizePre));
        MultiLabelOutput out = new MultiLabelOutput(outputs.toArray(), threshold);
	    return out;
	}

	private ArrayRealVector setupPredictKernel(Instance testInstance){
		ArrayRealVector dTestInstance = new ArrayRealVector(testInstance.toDoubleArray());
		int numTraining = this.SVs.getColumnDimension();
		ArrayRealVector kernel = new ArrayRealVector(numTraining);
		if (this.kType.equals(KernelType.RBF)){
	        for (int j = 0; j < numTraining; j++){
				ArrayRealVector powTemp = dTestInstance.subtract(
						SVs.getColumnVector(j)).mapToSelf(new Power(2));
				double exponent = -this.gamma * StatUtils.sum(powTemp.toArray());
				kernel.setEntry(j, FastMath.exp(exponent));
	        }
		}
		else if(this.kType.equals(KernelType.POLYNOMIAL)){
            for (int j = 0; j < numTraining; j++){
            	double dotProd = dTestInstance.dotProduct(SVs.getColumnVector(j));
            	double base = this.gamma * dotProd + this.coefficient;
            	double value = FastMath.pow(base, this.degree);
                kernel.setEntry(j, value);
            }
		}
	    else{
	    	RealMatrix instMat = MatrixUtils.createRowRealMatrix(dTestInstance.toArray());
	    	kernel = (ArrayRealVector) instMat.multiply(SVs).getRowVector(1);
	    }
		return kernel;
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
	
	private class ValueIndexPair implements Comparable<ValueIndexPair> {
	    public final int index;
	    public final double value;

	    ValueIndexPair(int index, double value) {
	        this.index = index;
	        this.value = value;
	    }

	    @Override
	    public int compareTo(ValueIndexPair other) {
	        return Double.valueOf(this.value).compareTo(other.value);
	    }
	}
}
