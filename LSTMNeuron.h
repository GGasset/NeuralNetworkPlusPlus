#include "ValueGeneration.h"
#include "NeuronConnectionsInfo.h"
#include "ActivationFunctions.h"

#pragma once
class LSTMNeuron
{
private:
	ActivationFunctions::ActivationFunction ActivationFunction;

public:
	NeuronConnectionsInfo connections;
	float ForgetGateWeight, StoreGateWeight, OutputGateWeight;

	LSTMNeuron(size_t layerI, size_t previousLayerLength, ActivationFunctions::ActivationFunction activationFunction, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		ActivationFunction = activationFunction;
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		ForgetGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		OutputGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
	}
};

