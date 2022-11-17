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

	void ApplyGradients(NeuronConnectionsInfo gradients, float learningRate)
	{
		for (size_t i = 0; weightIterator != Weights.end() && gWeightIterator != gradients.Weights.end(); i++, weightIterator++, gWeightIterator++)
		{
			(*weightIterator) -= (*gWeightIterator) * learningRate;
		}
	}



	size_t GetConnectionCount()
	{
		return connectionCount;
	}
};

