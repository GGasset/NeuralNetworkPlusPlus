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

		//Store gate
		output.StoreSigmoidWeightMultiplication = output.HiddenLinearSigmoid * StoreGateSigmoidWeight;
		output.StoreTanhWeightMultiplication = output.HiddenLinearTanh * StoreGateTanhWeight;
		output.StoreGateMultiplication = output.StoreSigmoidWeightMultiplication * output.StoreTanhWeightMultiplication;
		output.CellState = cellState = output.StoreGateAddition = output.StoreGateMultiplication + cellState;

		//Output Gate
		output.CellStateTanh = ActivationFunctions::TanhActivation(cellState);
		output.OutputWeightMultiplication = output.HiddenLinearSigmoid * OutputGateWeight;

		output.OutputActivation = output.HiddenState = output.CellStateTanh * output.OutputWeightMultiplication;

		return output;
	}

	LSTMNeuron* GetGradients(size_t tCount, NeuronStoredValues storedExecution, float neuronCost, float*** networkCosts, float*** networkActivations)
	{
		for (size_t t = 0; t < tCount; t++)
		{

		}
	}
};

