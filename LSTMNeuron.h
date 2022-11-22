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
		hiddenState = 0;
		cellState = 0;

		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
		ForgetGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateSigmoidWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		StoreGateTanhWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
		OutputGateWeight = ValueGeneration::GenerateWeight(minWeight, weightClosestTo0, maxWeight);
	}

	NeuronStoredValues ExecuteStore(float** networkActivations)
	{
		NeuronStoredValues output = NeuronStoredValues();
		output.InitialHiddenState = hiddenState;
		output.InitialCellState = cellState;

		output.LinearFunction = connections.LinearFunction(networkActivations);
		output.HiddenLinear = hiddenState + output.LinearFunction;

		output.HiddenLinearSigmoid = ActivationFunctions::SigmoidActivation(output.HiddenLinear);
		output.HiddenLinearTanh = ActivationFunctions::TanhActivation(output.HiddenLinear);


		//Forget gate

		output.ForgetWeightMultiplication = output.HiddenLinearSigmoid * ForgetGateWeight;
		cellState = output.ForgetGateMultiplication = output.ForgetWeightMultiplication * cellState;
	}
};

