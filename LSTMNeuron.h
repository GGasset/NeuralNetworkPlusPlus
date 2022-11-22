#include "ValueGeneration.h"
#include "NeuronConnectionsInfo.h"
#include "ActivationFunctions.h"
#include "NeuronStoredValues.h"

#pragma once
class LSTMNeuron
{
public:
	NeuronConnectionsInfo connections;
	float hiddenState, cellState;
	float ForgetGateWeight, StoreGateSigmoidWeight, StoreGateTanhWeight, OutputGateWeight;

	LSTMNeuron(size_t layerI, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		ForgetGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateSigmoidWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateTanhWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		OutputGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		hiddenState = 0;
		cellState = 0;
	}

	NeuronStoredValues ExecuteStore(float** networkActivations)
	{
		NeuronStoredValues output;
		output.LinearFunction = connections.LinearFunction(networkActivations);
		
	}
};

