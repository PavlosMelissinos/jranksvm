package mlpms;

import weka.core.Instance;
import weka.core.TechnicalInformation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

@SuppressWarnings("serial")
public class RankSVM extends MultiLabelLearnerBase{

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet)
			throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance)
			throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

}
