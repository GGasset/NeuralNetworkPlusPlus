#pragma once
using namespace std;
#include <list>
#include <tuple>
#include "ValueGeneration.h"

class NeuronConnectionsInfo
{
public:
	list<long> Xs;
	list<long> Ys;

	list<float> Weights;

	float Bias;

	NeuronConnectionsInfo(long layerI, long previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		Weights = ValueGeneration::GenerateWeigths(previousLayerLength, minWeight, weightClosestTo0, maxWeight);

		tuple<list<long>, list<long>> connectedPositions = ValueGeneration::GenerateConnectedPositions(layerI - 1, 0, previousLayerLength);
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
		auto xsIterator = Xs.begin();
		auto ysIterator = Ys.begin();
		auto weightsIterator = Weights.begin();

		for (long i = 0; weightsIterator != Weights.end(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			long x, y;
			x = *xsIterator;
			y = *ysIterator;
			float currentActivation = networkActivations[x][y];
			linearFunction += currentActivation * (*weightsIterator);
		}
		return linearFunction;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="activationGradient"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(weightGradients, previousActivationGradients)</returns>
	tuple<list<float>, list<float>> GetGradients(float activationGradient, float** networkActivations)
	{
		list<float> weightGradients, previousActivationsGradients;
		weightGradients = list<float>();
		previousActivationsGradients = list<float>();
		auto xsIterator = Xs.begin();
		auto ysIterator = Ys.begin();
		auto weightsIterator = Weights.begin();

		for (int i = 0; weightsIterator != Weights.end(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			weightGradients.push_back(activationGradient * networkActivations[*xsIterator][*ysIterator]);
			previousActivationsGradients.push_back(activationGradient * *weightsIterator);
		}

		tuple<list<float>, list<float>> output(weightGradients, previousActivationsGradients);
		return output;
	}

	void ApplyGradients(NeuronConnectionsInfo gradients, float learningRate)
	{
		auto weightIterator = Weights.begin();
		auto gWeightIterator = gradients.Weights.begin();

		for (long i = 0; weightIterator != Weights.end(); i++, weightIterator++, gWeightIterator++)
		{
			(*weightIterator) -= (*gWeightIterator) * learningRate;
		}
	}

	long GetConnectionCount()
	{
		return Weights.size();
	}
};

