#pragma once
#include "NeuronConnectionsInfo.h";

class Neuron
{
public:
	NeuronConnectionsInfo connections;

	Neuron(int layerI, int previousLayerI, double bias, double minWeight, double weightClosestTo0, double maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerI, bias, minWeight, weightClosestTo0, maxWeight);
	}

	Neuron() {

	}


};

