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

	list<double> Weights;

	double Bias;

	NeuronConnectionsInfo(long layerI, long previousLayerLength, double bias, double minWeight, double weightClosestTo0, double maxWeight)
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

	double LinearFunction(double** networkActivations)
	{
		double linearFunction = Bias;
		auto xsIterator = Xs.begin();
		auto ysIterator = Ys.begin();
		auto weightsIterator = Weights.begin();

		for (long i = 0; weightsIterator != Weights.end(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			long x, y;
			x = *xsIterator;
			y = *ysIterator;
			double weight = *weightsIterator;
			double currentActivation = networkActivations[x][y];
			double currentMultiplicationValue = currentActivation * weight;
			linearFunction += currentMultiplicationValue;
		}
		return linearFunction;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="activationGradient"></param>
	/// <param name="networkActivations"></param>
	/// <returns>tuple(weightGradients, previousActivationGradients)</returns>
	tuple<list<double>, list<double>> GetGradients(double activationGradient, double** networkActivations)
	{
		list<double> weightGradients, previousActivationsGradients;
		weightGradients = list<double>();
		previousActivationsGradients = list<double>();
		auto xsIterator = Xs.begin();
		auto ysIterator = Ys.begin();
		auto weightsIterator = Weights.begin();

		for (int i = 0; weightsIterator != Weights.end(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			weightGradients.push_back(activationGradient * networkActivations[*xsIterator][*ysIterator]);
			previousActivationsGradients.push_back(activationGradient * *weightsIterator);
		}

		tuple<list<double>, list<double>> output(weightGradients, previousActivationsGradients);
		return output;
	}

	void ApplyGradients(NeuronConnectionsInfo gradients, double learningRate)
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

