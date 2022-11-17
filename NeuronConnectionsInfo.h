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

	NeuronConnectionsInfo() {
		Bias = 1;
	}

	float LinearFunction(float** networkActivations)
	{
		float linearFunction = Bias;

		for (size_t i = 0; connectionCount; i++)
		{
			linearFunction += networkActivations[Xs[i]][Ys[i]] * Weights[i];
		}
		return linearFunction;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="activationGradient"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(weightGradients, previousActivationGradients)</returns>
	tuple<float*, float*> GetGradients(float activationGradient, float** networkActivations)
	{
		float* weightGradients = new float[connectionCount];
		float* previousActivationsGradients = new float[connectionCount];

		for (int i = 0; i < connectionCount; i++)
		{
			weightGradients[i] = activationGradient * networkActivations[Xs[i]][Ys[i]];
			previousActivationsGradients[i] = activationGradient * Weights[i];
		}

		tuple<float*, float*> output(weightGradients, previousActivationsGradients);
		return output;
	}

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
			threads[nThreads] = thread(std::ref(gradientApplyers[nThreads]), nThreads * connectionsPerThread, connectionsPerThread, learningRate);
		}

		for (size_t i = 0; i < totalThreads; i++)
		{
			threads[i].join();
		}
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

