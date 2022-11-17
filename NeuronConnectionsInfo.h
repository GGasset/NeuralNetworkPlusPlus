#pragma once
using namespace std;
#include <list>
#include <tuple>
#include "ValueGeneration.h"

class NeuronConnectionsInfo
{
private:
	size_t connectionCount;

public:
	size_t* Xs;
	size_t* Ys;

	float* Weights;

	float Bias;

	const size_t connectionsPerThread = 350;

	NeuronConnectionsInfo(size_t layerI, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		connectionCount = previousLayerLength;

		Weights = ValueGeneration::GenerateWeigths(previousLayerLength, minWeight, weightClosestTo0, maxWeight, connectionsPerThread);

		tuple<size_t*, size_t*> connectedPositions = ValueGeneration::GenerateConnectedPositions(layerI - 1, 0, previousLayerLength, connectionsPerThread);
		Xs = get<0>(connectedPositions);
		Ys = get<1>(connectedPositions);
		Bias = bias;
	}

	NeuronConnectionsInfo operator=(const NeuronConnectionsInfo& in)
	{
		NeuronConnectionsInfo out;
		out.connectionCount = in.connectionCount;
		out.Xs = in.Xs;
		out.Ys = in.Ys;
		out.Weights = in.Weights;
		out.Bias = in.Bias;
		return out;
	}

	NeuronConnectionsInfo() {
		Bias = 1;
		connectionCount = 0;
		Xs = Ys = NULL;
		Weights = NULL;
	}

	float LinearFunction(float** networkActivations)
	{
		size_t nThreads = connectionCount / connectionsPerThread;
		size_t remainingConnections = connectionCount % connectionsPerThread;
		bool isThereARemainingThread = remainingConnections > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		thread* threads = new thread[totalThreads];
		LinearFunctionCalculator* linearFunctionCalculators = new LinearFunctionCalculator[totalThreads];

		for (size_t i = 0; i < nThreads; i++)
		{
			threads[i] = thread(std::ref(linearFunctionCalculators[i]), this, networkActivations, connectionsPerThread * i, connectionsPerThread);
		}
		if (isThereARemainingThread)
			threads[nThreads] = thread(std::ref(linearFunctionCalculators[nThreads]), this, networkActivations, connectionsPerThread * nThreads, remainingConnections);

		float linearFunction = Bias;
		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
			linearFunction += linearFunctionCalculators[i].output;
		}

		delete[] threads;
		delete[] linearFunctionCalculators;

		return linearFunction;
	}

private:
	class LinearFunctionCalculator
	{
	public:
		float output;

		void operator()(NeuronConnectionsInfo* connectionsInfo, float** networkActivations, size_t startingI, size_t outputConnectionsCount)
		{
			output = 0;
			for (size_t i = startingI; i < startingI + outputConnectionsCount; i++)
			{
				output += networkActivations[connectionsInfo->Xs[i]][connectionsInfo->Ys[i]] * connectionsInfo->Weights[i];
			}
		}
	};

public:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="activationGradient"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(weightGradients, previousActivationGradients)</returns>
	tuple<float*, float*> GetGradients(float activationGradient, float** networkActivations)
	{
		size_t nThreads = connectionCount / connectionsPerThread;
		size_t remainingConnections = connectionCount % connectionsPerThread;
		bool isThereARemainingThread = remainingConnections > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		GradientCalculator* gradientCalculators = new GradientCalculator[totalThreads];
		thread* threads = new thread[totalThreads];

		float* weightGradients = new float[connectionCount];
		float* previousActivationsGradients = new float[connectionCount];
		for (size_t i = 0; i < nThreads; i++)
		{
			threads[i] = thread(std::ref(gradientCalculators[i]), this, activationGradient, networkActivations, weightGradients, previousActivationsGradients,
				connectionsPerThread * i, connectionsPerThread);
		}
		if (isThereARemainingThread)
		{
			threads[nThreads] = thread(std::ref(gradientCalculators[nThreads]), this, activationGradient, networkActivations, weightGradients, previousActivationsGradients,
				connectionsPerThread * nThreads, remainingConnections);
		}

		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
		}

		delete[] gradientCalculators;
		delete[] threads;

		tuple<float*, float*> output(weightGradients, previousActivationsGradients);
		return output;
	}

private:
	class GradientCalculator
	{
	public:
		void operator()(NeuronConnectionsInfo* neuronConnectionsInfo, float activationGradient, float** networkActivations, float* outputWeightGradients, float* outputPreviousActivationsGradients,
			size_t startingI, size_t connectionsToCalculate)
		{
			for (int i = startingI; i < startingI + connectionsToCalculate; i++)
			{
				outputWeightGradients[i] = activationGradient * networkActivations[neuronConnectionsInfo[0].Xs[i]][neuronConnectionsInfo[0].Ys[i]];
				outputPreviousActivationsGradients[i] = activationGradient * neuronConnectionsInfo[0].Weights[i];
			}
		}
	};

public:
	void ApplyGradients(NeuronConnectionsInfo& gradients, float learningRate)
	{
		size_t nThreads = connectionCount / connectionsPerThread;
		size_t leftConnections = connectionCount % connectionsPerThread;
		bool isThereARemainingThread = leftConnections > 0;
		size_t totalThreads = nThreads + isThereARemainingThread;

		GradientApplyer* gradientApplyers = new GradientApplyer[totalThreads];
		thread* threads = new thread[totalThreads];

		for (size_t i = 0; i < nThreads; i++)
		{
			gradientApplyers[i].Connections = this;
			gradientApplyers[i].Gradients = &gradients;
			threads[i] = thread(std::ref(gradientApplyers[i]), connectionsPerThread * i, connectionsPerThread, learningRate);
		}
		if (isThereARemainingThread)
		{
			gradientApplyers[nThreads].Connections = this;
			gradientApplyers[nThreads].Gradients = &gradients;
			threads[nThreads] = thread(std::ref(gradientApplyers[nThreads]), nThreads * connectionsPerThread, leftConnections, learningRate);
		}

		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
		}

		delete[] gradientApplyers;
		delete[] threads;
		gradients.Dispose();
	}

private:
	class GradientApplyer
	{
	public:
		NeuronConnectionsInfo* Connections;
		NeuronConnectionsInfo* Gradients;

		void operator()(size_t startingI, size_t weightsToApplyCount, float learningRate)
		{
			for (size_t i = 0; i < weightsToApplyCount; i++)
			{
				Connections[0].Weights[startingI + i] -= Gradients[0].Weights[startingI + i] * learningRate;
			}
		}
	};

public:
	size_t GetConnectionCount()
	{
		return connectionCount;
	}

	void Dispose()
	{
		delete[] Xs;
		delete[] Ys;
		delete[] Weights;
	}
};

