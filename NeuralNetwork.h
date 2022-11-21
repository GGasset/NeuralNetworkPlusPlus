using namespace std;
#include "ActivationFunctions.h";
#include "Cost.h"
#include "Neuron.h"
#include <list>
#include <tuple>
#include <cmath>

#pragma once
class NeuralNetwork
{
public:
	list<list<Neuron>> Neurons;
	ActivationFunctions::ActivationFunction ActivationFunction;
	size_t OutputLength;

public:
	/// <summary>
	/// Remember to .Dispose() this network
	/// </summary>
	NeuralNetwork(size_t shapeLength, size_t shape[], bool deleteShapeArr, ActivationFunctions::ActivationFunction activationFunction, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		LayerInstantiator* layerInstantiators = new LayerInstantiator[shapeLength - 1];
		thread* threads = new thread[shapeLength - 1];

		for (size_t i = 0; i < shapeLength - 1; i++)
		{
			threads[i] = thread(std::ref(layerInstantiators[i]), i + 1, shape[i + 1], shape[i], bias, minWeight, weightClosestTo0, maxWeight);
		}

		Neurons = list<list<Neuron>>();
		for (size_t i = 0; i < shapeLength - 1; i++)
		{
			threads[i].join();
			Neurons.push_back(layerInstantiators[i].instantiatedLayer);
		}

		OutputLength = shape[shapeLength - 1];

		ActivationFunction = activationFunction;

		delete[] layerInstantiators;
		delete[] threads;
		if (deleteShapeArr)
			delete[] shape;
	}

private:
	class LayerInstantiator
	{
	public:
		list<Neuron> instantiatedLayer;

		void operator()(size_t layerI, size_t layerLength, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
		{
			instantiatedLayer = GenerateNeuronLayer(layerI, layerLength, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		}
	};

	static list<Neuron> GenerateNeuronLayer(size_t layerI, size_t layerLength, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		list<Neuron> layer = list<Neuron>();
		NeuronInstatiator* neuronInstantiators = new NeuronInstatiator[layerLength];
		thread* neuronThreads = new thread[layerLength];

		for (size_t i = 0; i < layerLength; i++)
		{
			neuronThreads[i] = thread(std::ref(neuronInstantiators[i]), layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		}

		for (size_t i = 0; i < layerLength; i++)
		{
			neuronThreads[i].join();
			layer.push_front(neuronInstantiators[i].instantiatedNeuron);
		}

		delete[] neuronThreads;
		delete[] neuronInstantiators;
		return layer;
	}

	class NeuronInstatiator
	{
	public:
		Neuron instantiatedNeuron;

		void operator()(size_t layerI, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
		{
			instantiatedNeuron = Neuron(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		}
	};

	NeuralNetwork(list<list<Neuron>> neurons, ActivationFunctions::ActivationFunction activationFunction, size_t outputLength)
	{
		Neurons = neurons;
		ActivationFunction = activationFunction;
		OutputLength = outputLength;
	}

	NeuralNetwork()
	{
		ActivationFunction = ActivationFunctions::Sigmoid;
		OutputLength = 0;
	}

public:
	float* Execute(float* input)
	{
		tuple<float**, float**> storedExecution = ExecuteStore(input);
		float** linears = get<0>(storedExecution);
		float** activations = get<1>(storedExecution);

		float* output = activations[GetNetworkLayerCount()];

		delete[] linears[0];
		for (size_t i = 1; i < GetNetworkLayerCount(); i++)
		{
			delete[] linears[i];
			delete[] activations[i];
		}
		delete[] linears;
		delete[] activations;

		return output;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="input"></param>
	/// <returns>tuple(networkLinears, networkActivations)</returns>
	tuple<float**, float**> ExecuteStore(float* input)
	{
		float** networkLinearFunctions = new float* [GetNetworkLayerCount()];
		float** networkActivations = new float* [GetNetworkLayerCount() + 1];
		networkActivations[0] = input;
		auto layerIter = Neurons.begin();
		for (size_t i = 0; layerIter != Neurons.end(); i++, layerIter++)
		{
			tuple<float*, float*> layerExecutionResults = ExecuteStoreLayer(layerIter, networkActivations);
			networkLinearFunctions[i] = get<0>(layerExecutionResults);
			networkActivations[i + 1] = get<1>(layerExecutionResults);
		}

		tuple<float**, float**> output(networkLinearFunctions, networkActivations);
		return output;
	}

private:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="layerIter"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(linearFunctions, networkActivations)</returns>
	tuple<float*, float*> ExecuteStoreLayer(list<list<Neuron>>::iterator layerIter, float** networkActivations)
	{
		list<Neuron> layer = (*layerIter);
		size_t layerLength = layer.size();
		float* layerLinears = new float[layerLength];
		float* layerActivations = new float[layerLength];

		auto neuronIter = layer.begin();
		thread* threads = new thread[layerLength];
		NeuronExecutor* executors = new NeuronExecutor[layerLength];

		for (size_t i = 0; neuronIter != layer.end(); i++, neuronIter++)
		{
			threads[i] = thread(std::ref(executors[i]), neuronIter, networkActivations, ActivationFunction);
		}

		for (size_t i = 0; i < layerLength; i++)
		{
			threads[i].join();
			layerLinears[i] = executors[i].LinearFunction;
			layerActivations[i] = executors[i].Activation;
		}

		delete[] threads;
		delete[] executors;

		tuple<float*, float*> layerExecutionResults(layerLinears, layerActivations);
		return layerExecutionResults;
	}

	class NeuronExecutor
	{
	public:
		float LinearFunction;
		float Activation;

		void operator()(list<Neuron>::iterator neuronIter, float** networkActivations, ActivationFunctions::ActivationFunction activationFunction)
		{
			Neuron neuron = (*neuronIter);
			tuple<float, float> neuronOutput = neuron.ExecuteStore(networkActivations, activationFunction);
			LinearFunction = get<0>(neuronOutput);
			Activation = get<1>(neuronOutput);
		}
	};

public:
	/// <param name="trainUntilTestCostIsBelow">Put a high value to iterate once over all data</param>
	void SupervisedTrain(float** X, float** Y, size_t dataLength, size_t batchSize, Cost::CostFunction costFunction, float learningRate, bool freeData, double testSize = 0.2, double trainUntilTestCostIsBelow = .15)
	{
		size_t trainSize = std::fmaxf(0, std::fminf(1, testSize));
		trainSize = 1 - testSize;

		DataManipulation::ShuffleData(X, dataLength);
		DataManipulation::ShuffleData(Y, dataLength);

		tuple<float**, float**, size_t> slicedX = DataManipulation::SliceData(X, dataLength, trainSize, freeData);
		float** trainX = get<0>(slicedX);
		float** testX = get<1>(slicedX);

		tuple<float**, float**, size_t> slicedY = DataManipulation::SliceData(Y, dataLength, trainSize, freeData);
		float** trainY = get<0>(slicedY);
		float** testY = get<1>(slicedY);

		size_t trainDataLength = min(get<2>(slicedX), get<2>(slicedY));
		size_t testDataLength = dataLength - max(get<2>(slicedX), get<2>(slicedY));

		float testCost = 50E30f;
		size_t batchCount = trainDataLength / batchSize;
		size_t lastBatchSize = trainDataLength % batchSize;
		while (testCost > trainUntilTestCostIsBelow)
		{
			for (size_t i = 0; i < batchCount; i++)
			{
				SupervisedLearningBatch(trainX, trainY, batchSize * i, batchSize, costFunction, learningRate);
			}
			SupervisedLearningBatch(trainX, trainY, batchCount * batchSize, lastBatchSize, costFunction, learningRate);

			testCost = 0;
			for (size_t i = 0; i < testDataLength; i++)
			{
				testCost += Cost::GetCostOf(OutputLength, Execute(testX[i]), testY[i], costFunction);
			}
			testCost /= testDataLength;
		}
	}

	void SupervisedLearningBatch(float** X, float** Y, size_t startingIndex, size_t batchLength, Cost::CostFunction costFunction, float learningRate)
	{
		thread* threads = new thread[batchLength];
		NetworkGradientsCalculator* gradientCalculators = new NetworkGradientsCalculator[batchLength];

		for (size_t i = 0; i < batchLength; i++)
		{
			int dataI = startingIndex + i;
			gradientCalculators[i].network = this;
			threads[i] = thread(std::ref(gradientCalculators[i]), X[dataI], Y[dataI], costFunction);
		}

		NeuralNetwork* gradients = new NeuralNetwork[batchLength];
		for (size_t i = 0; i < batchLength; i++)
		{
			threads[i].join();
			gradients[i] = (*gradientCalculators[i].network);
		}

		ApplyGradients(gradients, batchLength, learningRate);

		delete[] threads;
		delete[] gradientCalculators;
		delete[] gradients;
	}

private:
	class NetworkGradientsCalculator
	{
	public:
		NeuralNetwork* network;
		NeuralNetwork* gradients;

		void operator()(float* X, float* Y, Cost::CostFunction costFunction)
		{
			(*gradients) = (*network).GetGradients(X, Y, costFunction);
		}
	};
	
public:
	NeuralNetwork GetGradients(float* X, float* Y, Cost::CostFunction costFunction)
	{
		tuple<float**, float**> executionResults = ExecuteStore(X);
		float** networkLinears, **networkActivations;
		networkLinears = get<0>(executionResults);
		networkActivations = get<1>(executionResults);

		float* cost = Derivatives::DerivativeOf(GetNetworkOutputLength(), get<1>(executionResults)[GetNetworkLayerCount()], Y, costFunction);
		NeuralNetwork gradients = GetGradients(networkLinears, networkActivations, cost);

		delete[] networkLinears[0];
		for (size_t i = 1; i < GetNetworkLayerCount(); i++)
		{
			delete[] networkLinears[i];
			delete[] networkActivations[i];
		}
		delete[] networkActivations[GetNetworkLayerCount()];

		delete[] networkLinears;
		delete[] networkActivations;

		return gradients;
	}

	NeuralNetwork GetGradients(float** networkLinears, float** networkActivations, float* costGradients)
	{
		auto layerIterator = Neurons.begin();
		float** neuronCosts = new float* [GetNetworkLayerCount() + 1];

		neuronCosts[GetNetworkLayerCount()] = costGradients;
		for (size_t i = 0; i < GetNetworkLayerCount(); i++)
		{
			neuronCosts[i] = new float[(*layerIterator).size()];
		}

		size_t layerI = GetNetworkLayerCount();
		list<list<Neuron>> gradientLayers = list<list<Neuron>>();
		layerIterator = Neurons.end();
		do
		{
			layerIterator--;

			gradientLayers.push_front(CalculateLayerGradients(layerI, &layerIterator, neuronCosts, networkLinears, networkActivations));

			layerI--;
		} while (layerIterator != Neurons.begin());

		NeuralNetwork gradientsNetwork = NeuralNetwork(gradientLayers, ActivationFunction, OutputLength);
		return gradientsNetwork;
	}

private:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="layerI">Takes into account input layer as index 0</param>
	/// <param name="layerIterator"></param>
	/// <param name="networkCosts"></param>
	/// <param name="networkLinears"></param>
	/// <param name="networkActivations"></param>
	/// <returns></returns>
	list<Neuron> CalculateLayerGradients(size_t layerI, list<list<Neuron>>::iterator* layerIterator, float** networkCosts, float** networkLinears, float** networkActivations)
	{
		list<Neuron> layer = (**layerIterator);
		size_t layerLength = layer.size();

		NeuronGradientsCalculator* gradientCalculators = new NeuronGradientsCalculator[layerLength];
		thread* threads = new thread[layerLength];
		list<Neuron>::iterator* neuronIterators = new list<Neuron>::iterator[layerLength];

		auto neuronIterator = layer.begin();
		size_t i = 0;
		while (neuronIterator != layer.end())
		{
			float linearFunction = networkLinears[layerI - 1][i];
			float cost = networkCosts[layerI][i];

			neuronIterators[i] = neuronIterator;

			threads[i] = thread(std::ref(gradientCalculators[i]), &neuronIterators[i], networkCosts, networkActivations, linearFunction, cost, ActivationFunction);

			neuronIterator++;
			i++;
		}

		list<Neuron> gradientsLayer = list<Neuron>();
		for (i = 0; i < layerLength; i++)
		{
			threads[i].join();
			gradientsLayer.push_back(gradientCalculators[i].outputGradients);
		}

		delete[] gradientCalculators;
		delete[] threads;
		delete[] neuronIterators;

		return gradientsLayer;
	}

	class NeuronGradientsCalculator
	{
	public:
		Neuron outputGradients;

		void operator()(list<Neuron>::iterator* neuronIterator, float** networkCosts, float** networkActivations, float linearFunction, float neuronCost, ActivationFunctions::ActivationFunction activationFunction)
		{
			tuple<float, float*, float*> gradients = (**neuronIterator).GetGradients(networkActivations, linearFunction, neuronCost, activationFunction);
			outputGradients = Neuron(get<0>(gradients), new size_t[0], new size_t[0], get<1>(gradients));

			NeuronConnectionsInfo* neuronInfo = &(*neuronIterator)->connections;
			float* previousActivationsGradients = get<2>(gradients);
			for (size_t j = 0; j < neuronInfo[0].GetConnectionCount(); j++)
			{
				networkCosts[neuronInfo[0].Xs[j]][neuronInfo[0].Ys[j]] -= previousActivationsGradients[j];
			}
			delete[] previousActivationsGradients;
		}
	};

public:
	void ApplyGradients(NeuralNetwork* gradients, size_t networkGradientsCount, float learningRate)
	{
		thread* threads = new thread[networkGradientsCount];
		NetworkGradientsApplyer* gradientApplyers = new NetworkGradientsApplyer[networkGradientsCount];

		for (size_t i = 0; i < networkGradientsCount; i++)
		{
			gradientApplyers[i].network = this;
			gradientApplyers[i].gradients = gradients + i;
		}
	}

private:
	class NetworkGradientsApplyer
	{
	public:
		NeuralNetwork* network;
		NeuralNetwork* gradients;

		void operator()(float learningRate)
		{
			(*network).ApplyGradients(*gradients, learningRate);
		}
	};

public:
	void ApplyGradients(NeuralNetwork gradients, float learningRate)
	{
		auto layerIterator = Neurons.begin();
		auto gLayerIterator = gradients.Neurons.begin();
		for (size_t i = 0; layerIterator != Neurons.end(); i++, layerIterator++, gLayerIterator++)
		{
			auto neuronIterator = (*layerIterator).begin();
			auto gNeuronIterator = (*gLayerIterator).begin();
			for (size_t i = 0; neuronIterator != (*layerIterator).end(); i++, neuronIterator++, gNeuronIterator++)
			{
				(*neuronIterator).ApplyGradients((*gNeuronIterator), learningRate);
			}
		}
	}


	ActivationFunctions::ActivationFunction GetActivationFunction()
	{
		return ActivationFunction;
	}

	/// <summary>
	/// Output includes input layer
	/// </summary>
	/// <returns></returns>
	list<size_t> GetNetworkShape()
	{
		size_t networkLength = GetNetworkLayerCount();
		list<size_t> shape = list<size_t>();
		shape.push_back(GetNetworkInputLength());

		auto layerIterator = Neurons.begin();
		for (size_t i = 0; layerIterator != Neurons.end(); i++, layerIterator++)
		{
			shape.push_back((*layerIterator).size());
		}
		return shape;
	}

	/// <summary>
	/// Output doesn't include input layer
	/// </summary>
	/// <returns></returns>
	size_t GetNetworkLayerCount()
	{
		return Neurons.size();
	}

	size_t GetNetworkInputLength()
	{
		auto layerIterator = Neurons.begin();
		auto neuronIterator = (*layerIterator).begin();
		return (*neuronIterator).connections.GetConnectionCount();
	}

	size_t GetNetworkOutputLength()
	{
		return OutputLength;
	}

	void Dispose()
	{
		auto layerIterator = Neurons.begin();
		while (layerIterator != Neurons.end())
		{
			auto neuronIterator = (*layerIterator).begin();
			while (neuronIterator != (*layerIterator).end())
			{
				(*neuronIterator).connections.Dispose();

				neuronIterator++;
			}

			layerIterator++;
		}
	}
};

