#pragma once
using namespace std;
#include <list>
#include "ValueGeneration.h"
#include <tuple>

class NeuronConnectionsInfo
{
public:
	list<int> Xs;
	list<int> Ys;

	list<double> Weights;

	NeuronConnectionsInfo(int layerI, int previousLayerLength, double minWeight, double weightClosestTo0, double maxWeight)
	{
		Weights = ValueGeneration::GenerateWeigths(previousLayerLength, minWeight, weightClosestTo0, maxWeight);

		tuple<list<int>, list<int>> connectedPositions = ValueGeneration::GenerateConnectedPositions(layerI, 0, previousLayerLength);
		Xs = get<0>(connectedPositions);
		Ys = get<1>(connectedPositions);
	}

	tuple<list<double>, list<double>> GetGradients(double biasGrad, double** previousActivations)
	{

	}
};

