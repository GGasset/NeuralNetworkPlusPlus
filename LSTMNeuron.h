#include "INeuron.h"

#pragma once
class LSTMNeuron : public INeuron
{
public:
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

	float Execute(float** networkActivations, ActivationFunctions::ActivationFunction activationType = ActivationFunctions::None)
	{
		NeuronStoredValues storedExecution = RecurrentExecuteStore(networkActivations, activationType);
		return storedExecution.OutputActivation;
	}

	NeuronStoredValues RecurrentExecuteStore(float** networkActivations, ActivationFunctions::ActivationFunction activationType = ActivationFunctions::None)
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

	tuple<NeuronConnectionsInfo*, float**> GetRecurrentGradients(size_t tCount, NeuronStoredValues storedExecution, float* neuronCosts, float*** networkCosts, float*** networkActivations)
	{
		NeuronStoredValues* derivatives = new NeuronStoredValues[tCount];
		std::thread* threads = new std::thread[tCount];

		// Get Derivatives
		for (size_t t = 0; t < tCount; t++)
		{
			
		}

		delete[] derivatives;
		tuple<NeuronConnectionsInfo*, float**> output;
		return output;
	}

private:
	NeuronStoredValues GetDerivatives(NeuronStoredValues executionResults)
	{
		NeuronStoredValues derivatives = NeuronStoredValues();

		derivatives.HiddenLinearSigmoid = Derivatives::SigmoidDerivative(executionResults.HiddenLinear);

		// Forget Gate

		derivatives.ForgetWeightMultiplication = executionResults.HiddenLinearSigmoid * derivatives.HiddenLinearSigmoid;

		derivatives.ForgetGateMultiplication =
			Derivatives::MultiplicationDerivative
			(
				executionResults.ForgetWeightMultiplication, executionResults.InitialCellState,
				derivatives.ForgetWeightMultiplication, 1
			);

		// Store Gate

		derivatives.HiddenLinearTanh = Derivatives::TanhDerivative(executionResults.HiddenLinear);

		derivatives.StoreSigmoidWeightMultiplication = derivatives.ForgetWeightMultiplication;

		derivatives.StoreTanhWeightMultiplication = executionResults.HiddenLinearTanh * derivatives.HiddenLinearTanh;

		derivatives.StoreGateMultiplication =
			Derivatives::MultiplicationDerivative
			(
				executionResults.StoreSigmoidWeightMultiplication, executionResults.StoreTanhWeightMultiplication,
				derivatives.StoreSigmoidWeightMultiplication, derivatives.StoreTanhWeightMultiplication
			);
		
		derivatives.StoreGateAddition = derivatives.ForgetGateMultiplication + derivatives.StoreGateMultiplication;

		// Output Gate

		derivatives.CellStateTanh = Derivatives::TanhDerivative(executionResults.CellState);

		derivatives.OutputWeightMultiplication = derivatives.ForgetWeightMultiplication;

		derivatives.OutputWeightMultiplication = 
			Derivatives::MultiplicationDerivative
			(
				executionResults.HiddenLinearSigmoid, executionResults.CellStateTanh, 
				derivatives.OutputWeightMultiplication, derivatives.CellStateTanh
			);
	}
};

