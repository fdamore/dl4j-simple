package org.cnr.simple;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * "Saturn" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierSaturn {

	public static void main(final String[] args) throws Exception {
		// Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
		final int batchSize = 50;
		final int seed = 123;
		final double learningRate = 0.005;
		// Number of epochs (full passes of the data)
		final int nEpochs = 30;

		final int numInputs = 2;
		final int numOutputs = 2;
		final int numHiddenNodes = 20;

		final String filenameTrain = new ClassPathResource("/classification/saturn_data_train.csv").getFile().getPath();
		final String filenameTest = new ClassPathResource("/classification/saturn_data_eval.csv").getFile().getPath();

		// Load the training data:
		final RecordReader rr = new CSVRecordReader();
		final FileSplit split = new FileSplit(new File(filenameTrain));
		rr.initialize(split);
		final DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

		// Load the test/evaluation data:
		final RecordReader rrTest = new CSVRecordReader();
		rrTest.initialize(new FileSplit(new File(filenameTest)));
		final DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

		// updater
		final Nesterovs nesterovs = new Nesterovs(0.9);
		nesterovs.setLearningRate(learningRate);

		// log.info("Build model....");
		final NeuralNetConfiguration.Builder neuralnet_builder = new NeuralNetConfiguration.Builder();

		// configure neural network params...
		neuralnet_builder.seed(seed);
		neuralnet_builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

		neuralnet_builder.updater(nesterovs);

		// configure dense layer - INPUT
		final DenseLayer.Builder dense_layer_builder = new DenseLayer.Builder();
		dense_layer_builder.nIn(numInputs);
		dense_layer_builder.nOut(numHiddenNodes);
		dense_layer_builder.weightInit(WeightInit.XAVIER);
		dense_layer_builder.activation(Activation.RELU);

		// configure OUTPUT LAYER
		final OutputLayer.Builder output_layer_builder = new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD);
		output_layer_builder.weightInit(WeightInit.XAVIER);
		output_layer_builder.activation(Activation.SOFTMAX);
		output_layer_builder.nIn(numHiddenNodes);
		output_layer_builder.nOut(numOutputs);

		// define layers
		final ListBuilder list_builder = neuralnet_builder.list();
		list_builder.layer(0, dense_layer_builder.build());
		list_builder.layer(1, output_layer_builder.build());
		list_builder.pretrain(false);
		list_builder.backprop(true);

		final MultiLayerConfiguration conf = list_builder.build();

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

		System.out.println(eval.stats());

		System.out.println("****************Example finished********************");
	}

}
