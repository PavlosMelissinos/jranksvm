package mlpms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.stream.DoubleStream;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

import org.apache.commons.math.FunctionEvaluationException;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.stat.descriptive.summary.Sum;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.PointVectorValuePair;
import org.apache.commons.math3.optim.linear.LinearConstraint;
import org.apache.commons.math3.optim.linear.LinearConstraintSet;
import org.apache.commons.math3.optim.linear.LinearObjectiveFunction;
import org.apache.commons.math3.optim.linear.LinearOptimizer;
import org.apache.commons.math3.optim.linear.NonNegativeConstraint;
import org.apache.commons.math3.optim.linear.Relationship;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.optimization.linear.*;

import scpsolver.constraints.LinearBiggerThanEqualsConstraint;
import scpsolver.constraints.LinearSmallerThanEqualsConstraint;
import scpsolver.lpsolver.LinearProgramSolver;
import scpsolver.lpsolver.SolverFactory;
import scpsolver.problems.LinearProgram;
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

	enum KernelType {
		LINEAR, POLYNOMIAL, RBF;
	}

	private KernelType kType;

	private double coefficient;

	/** Train labels. */
	// protected double[] m_class;
	private double[][] support_vectors;

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
		findAlpha(sizeAlpha, labelSize, Label, notLabel, kernel);
		// computeBias();
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

	private void findAlpha(ArrayRealVector sizeAlpha,
			ArrayRealVector labelSize, ArrayList<ArrayRealVector> Label,
			ArrayList<ArrayRealVector> notLabel, BlockRealMatrix kernel) {
		boolean continuing = true;

		ArrayList<BlockRealMatrix> cValue = trainingChunk1();

		int sizeAlphaSum = (int) DoubleStream.of(sizeAlpha.toArray())
				.map(Math::round).sum();
		this.alpha = new ArrayRealVector(sizeAlphaSum, 0);
		int numClass = this.trainTarget.getRowDimension();
		int numTraining = this.trainTarget.getColumnDimension();
		for (int iteration = 1; continuing; iteration++) {
			System.out.println("current iteration: " + iteration);

			// computeBeta(sizeAlpha, labelSize, Label, notLabel);
			BlockRealMatrix beta = new BlockRealMatrix(numClass, numTraining);
			for (int k = 0; k < numClass; k++) {
				for (int i = 0; i < numTraining; i++) {
					double sum = i > 0 ? StatUtils.sum(sizeAlpha.getSubVector(
							0, i).toArray()) : 0;
					assert labelSize.getEntry(i) == Label.get(i).getDimension();
					assert numClass - labelSize.getEntry(i) == notLabel.get(i)
							.getDimension();
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

			// computeGradient

			BlockRealMatrix inner = beta.multiply(kernel); // inner = beta X
															// kernel

			ArrayRealVector gradient = new ArrayRealVector(sizeAlphaSum);
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

			// findAlphaNew();
			BlockRealMatrix Aeq = new BlockRealMatrix(numClass, sizeAlphaSum);
			for (int k = 0; k < numClass; k++) {
				int counter = -1;
				for (int i = 0; i < numTraining; i++) {
					for (int m = 0; m < labelSize.getEntry(i); m++) {
						for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
							counter++;
							double temp1 = Label.get(i).getEntry(m);
							double temp2 = notLabel.get(i).getEntry(n);
							double temp = cValue.get(k).getEntry((int) temp1,
									(int) temp2);
							Aeq.addToEntry(k, counter, temp);
						}
					}
				}
			}
			ArrayRealVector beq = new ArrayRealVector(numClass);
			ArrayRealVector LB = new ArrayRealVector(sizeAlphaSum);
			ArrayRealVector UB = new ArrayRealVector(sizeAlphaSum);
			int counter = -1;
			for (int i = 0; i < numTraining; i++) {
				for (int m = 0; m < labelSize.getEntry(i); m++) {
					for (int n = 0; n < numClass - labelSize.getEntry(i); n++) {
						counter++;
						double temp = cost / sizeAlpha.getEntry(i);
						UB.addToEntry(counter, temp);
					}
				}
	}
	
	double findLambda(ArrayRealVector Alpha_new, ArrayList<BlockRealMatrix> cValue, 
			BlockRealMatrix kernel, int numTraining, int numClass, ArrayList<ArrayRealVector> Label, 
			ArrayList<ArrayRealVector> notLabel, ArrayRealVector labelSize, ArrayRealVector sizeAlpha){
	    NegDualFuncUniWrapper f = new NegDualFuncUniWrapper(alpha, Alpha_new, cValue, kernel, 
	    		numTraining, numClass, Label, notLabel, labelSize, sizeAlpha);
	    BrentOptimizer solver = new BrentOptimizer(1e-10, 1e-14);

		UnivariatePointValuePair solution = solver.optimize(new MaxEval(200), 
				new UnivariateObjectiveFunction(f), GoalType.MINIMIZE, new SearchInterval(0, 1));
		double lambda = solution.getPoint();
		return lambda;
			}

			System.out.println("Lilia: ");

			// find alpha new
			double[] gradientArray = gradient.toArray();
			double[][] AeqArray = Aeq.getData();
			double[] UBArray = UB.toArray();
			double[] LBArray = LB.toArray();
			double[] beqArray = beq.toArray();
			LinearObjectiveFunction f = new LinearObjectiveFunction(
					gradientArray, 0);
			Collection<LinearConstraint> constraints = new ArrayList<LinearConstraint>();
			// Constraint Aeq*x=beq
			for (int i = 0; i < Aeq.getRowDimension(); i++) {
				double[] Av = new double[Aeq.getColumnDimension()];
				for (int j = 0; j < Aeq.getColumnDimension(); j++) {
					Av[j] = AeqArray[i][j];
				}
				constraints.add(new LinearConstraint(Av, Relationship.EQ,
						beqArray[i]));

			}

			// Constraints x=<UB, x>=LB
			ArrayRealVector coeffTemp = new ArrayRealVector(
					Aeq.getColumnDimension());
			for (int k = 0; k < Aeq.getColumnDimension(); k++) {
				coeffTemp.setEntry(k, 0);
			}
			for (int j = 0; j < Aeq.getColumnDimension(); j++) {
				ArrayRealVector coeff = coeffTemp;
				coeff.setEntry(j, 1);
				constraints.add(new LinearConstraint(coeff, Relationship.LEQ,
						UBArray[j]));
				constraints.add(new LinearConstraint(coeff, Relationship.GEQ,
						LBArray[j]));
				coeff.setEntry(j, 0);
				System.out.println("Iteration: " + j);
			}

			SimplexSolver solver = new SimplexSolver();
			double[] solution = new double[sizeAlphaSum];
			PointValuePair optSolution = solver.optimize(new MaxIter(100), f,
					new LinearConstraintSet(constraints), GoalType.MINIMIZE);
			solution = optSolution.getPoint();
			System.out.println("Ok: ");
		}

	}

	private void computeBias() {
		// stub
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
