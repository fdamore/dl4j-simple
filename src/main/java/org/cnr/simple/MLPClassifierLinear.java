package org.cnr.simple;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierLinear {

	public static void main(final String[] args) throws Exception {
		final int seed = 123;
		final double learningRate = 0.01;
		final int batchSize = 50;
		final int nEpochs = 30;

		final int numInputs = 2;
		final int numOutputs = 2;
		final int numHiddenNodes = 20;

		final String filenameTrain = new ClassPathResource("/classification/linear_data_train.csv").getFile().getPath();
		final String filenameTest = new ClassPathResource("/classification/linear_data_eval.csv").getFile().getPath();

		// Load the training data:
		final RecordReader rr = new CSVRecordReader();
		// rr.initialize(new FileSplit(new
		// File("src/main/resources/classification/linear_data_train.csv")));
		rr.initialize(new FileSplit(new File(filenameTrain)));
		final DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

		// Load the test/evaluation data:
		final RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		final DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

		final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).updater(new Nesterovs(learningRate, 0.9)).list()
				.layer(0,
						new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation(Activation.RELU)
								.build())
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX)
						.nIn(numHiddenNodes).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build();

		final MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10)); // Print score every 10 parameter updates

		model.fit(trainIter, nEpochs);

		System.out.println("Evaluate model....");
		final Evaluation eval = new Evaluation(numOutputs);
		while (testIter.hasNext()) {
			final DataSet t = testIter.next();
			final INDArray features = t.getFeatureMatrix();
			final INDArray lables = t.getLabels();
			final INDArray predicted = model.output(features, false);

			eval.eval(lables, predicted);

		}

		// Print the evaluation statistics
		System.out.println(eval.stats());

		// ------------------------------------------------------------------------------------
		// Training is complete. Code that follows is for plotting the data &
		// predictions only

		System.out.println("****************Example finished********************");
	}
}