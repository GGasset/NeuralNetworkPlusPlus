#pragma once
using namespace std;
#include <list>
#include <tuple>
#include "ValueGeneration.h"

class NeuronConnectionsInfo
{
public:
	list<int> Xs;
	list<int> Ys;

	list<double> Weights;

	double Bias;

	NeuronConnectionsInfo(int layerI, int previousLayerLength, double bias, double minWeight, double weightClosestTo0, double maxWeight)
	{
		Weights = ValueGeneration::GenerateWeigths(previousLayerLength, minWeight, weightClosestTo0, maxWeight);

		tuple<list<int>, list<int>> connectedPositions = ValueGeneration::GenerateConnectedPositions(layerI, 0, previousLayerLength);
		Xs = get<0>(connectedPositions);
		Ys = get<1>(connectedPositions);
		Bias = bias;
	}

	NeuronConnectionsInfo() {
		Bias = 1;
	}

	double Execute(double** neuronsActivations)
	{
		double output = Bias;
		auto xsIterator = Xs.cbegin();
		auto ysIterator = Ys.cbegin();
		auto weightsIterator = Weights.cbegin();

		for (long i = 0; i < GetConnectionsLength(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			output += neuronsActivations[*xsIterator][*ysIterator] * *weightsIterator;
		}
		return output;
	}

	tuple<list<double>, list<double>> GetGradients(double biasGrad, double** networkActivations)
	{
		list<double> weightGradients, previousActivationsGradients;
		weightGradients = list<double>();
		previousActivationsGradients = list<double>();
		auto xsIterator = Xs.cbegin();
		auto ysIterator = Ys.cbegin();
		auto weightsIterator = Weights.cbegin();

		for (int i = 0; i < GetConnectionsLength(); i++, xsIterator++, ysIterator++, weightsIterator++)
		{
			weightGradients.push_back(biasGrad * networkActivations[*xsIterator][*ysIterator]);
			previousActivationsGradients.push_back(biasGrad * *weightsIterator);
		}

		tuple<list<double>, list<double>> output(weightGradients, previousActivationsGradients);
		return output;
	}

	int GetConnectionsLength()
	{
		return Weights.size();
	}
};

