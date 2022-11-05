#pragma once
#include "NeuronConnectionsInfo.h";
#include "ActivationFunctions.h"

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

	double Execute(double** neuronActivations, ActivationFunctions::ActivationFunction activationFunction)
	{
		return ActivationFunctions::Activate(connections.Execute(neuronActivations), activationFunction);
	}
};

