#include "ValueGeneration.h"
#include "NeuronConnectionsInfo.h"
#include "ActivationFunctions.h"

#pragma once
class LSTMNeuron
{
public:
	NeuronConnectionsInfo connections;
	float ForgetGateWeight, StoreGateSigmoidWeight, StoreGateTanhWeight, OutputGateWeight;

	LSTMNeuron(size_t layerI, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		ForgetGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateSigmoidWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateTanhWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		OutputGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
	}
};

