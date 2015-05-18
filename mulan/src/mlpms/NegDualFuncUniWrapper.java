package mlpms;

import java.util.ArrayList;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.stat.StatUtils;

public class NegDualFuncUniWrapper implements UnivariateFunction {
	private ArrayRealVector Alpha_old;
	private ArrayRealVector Alpha_new; 
	private ArrayList<BlockRealMatrix> c_value;
	private BlockRealMatrix kernel;
	private int num_training;
	private int num_class; 
	private ArrayList<ArrayRealVector> Label;
	private ArrayList<ArrayRealVector> not_Label;
	private ArrayRealVector Label_size;
	private ArrayRealVector size_alpha;
	
	public NegDualFuncUniWrapper(ArrayRealVector Alpha_old, ArrayRealVector Alpha_new, 
			ArrayList<BlockRealMatrix> c_value, BlockRealMatrix kernel, int num_training, int num_class, 
			ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> not_Label, ArrayRealVector Label_size, ArrayRealVector size_alpha){
		this.Alpha_old = Alpha_old;
		this.Alpha_new = Alpha_new;
		this.c_value = c_value;
		this.kernel = kernel;
		this.num_training = num_training;
		this.num_class = num_class;
		this.Label = Label;
		this.not_Label = not_Label;
		this.Label_size = Label_size;
		this.size_alpha = size_alpha;
	}
	
	@Override
	public double value(double Lambda) {
		// TODO Auto-generated method stub
		return neg_dual_func(Lambda, Alpha_old, Alpha_new, c_value, kernel, num_training, num_class, 
				Label, not_Label, Label_size, size_alpha);
	}

	private double neg_dual_func(double Lambda, ArrayRealVector Alpha_old, ArrayRealVector Alpha_new, 
			ArrayList<BlockRealMatrix> c_value, BlockRealMatrix kernel, int num_training, int num_class, 
			ArrayList<ArrayRealVector> Label, ArrayList<ArrayRealVector> not_Label, ArrayRealVector Label_size, ArrayRealVector size_alpha){
//		function output=neg_dual_func(Lambda,Alpha_old,Alpha_new,c_value,kernel,num_training,num_class,Label,not_Label,Label_size,size_alpha);

		ArrayRealVector alphaDiff = Alpha_new.subtract(Alpha_old);
		ArrayRealVector Alpha = Alpha_old.add(alphaDiff.mapMultiplyToSelf(Lambda));
	    //Beta=zeros(num_class,num_training);
		BlockRealMatrix Beta = new BlockRealMatrix(num_class, num_training);
		
		for (int k = 0; k < num_class; k++){
			for (int i = 0; i < num_training; i++){
				double sum = i > 0 ? StatUtils.sum(size_alpha.getSubVector(0, i).toArray()) : 0;
				for (int m = 0; m < Label_size.getEntry(i); m++){
					for (int n = 0; n < num_class - Label_size.getEntry(i); n++){
						int index = new Double(sum + m * (num_class - Label_size.getEntry(i)) + n + 1).intValue();
						//System.out.println(k + " + " + i + " + " + m + " + " + n + " + " + index);
						double oldBetaVal = Beta.getEntry(k, i);
						double alphaVal = Alpha.getEntry(index - 1);
						int labelIndex = new Double(Label.get(i).getEntry(m)).intValue();
						int notLabelIndex = new Double(not_Label.get(i).getEntry(n)).intValue();
						double cv = c_value.get(k).getEntry(labelIndex, notLabelIndex);
						double newBetaVal = oldBetaVal + cv * alphaVal;
						Beta.setEntry(k, i, newBetaVal);
					}
				}
			}
		}
		//Instead of computing the result for each row we do the full composition B * k * B'
		//and then get the trace (sum of main diagonal elements).
		
		BlockRealMatrix out = Beta.multiply(kernel.multiply(Beta.transpose()));
		double output = out.getTrace();
		output *= 0.5;
		output = output - StatUtils.sum(Alpha.toArray());
	    return output;
	}

}
