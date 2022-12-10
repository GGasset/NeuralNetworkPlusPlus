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

	tuple<NeuronConnectionsInfo*, float**> GetRecurrentGradients(size_t tCount, NeuronStoredValues* storedExecution, float* neuronCosts, float*** networkCosts, float*** networkActivations)
	{
		std::thread* threads = new std::thread[tCount];

		// Get Derivatives
		NeuronStoredValues* derivatives = new NeuronStoredValues[tCount];
		DerivativeCalculator* derivativeCalculators = new DerivativeCalculator[tCount];
		for (size_t t = 0; t < tCount; t++)
		{
			threads[t] = thread(std::ref(derivativeCalculators[t]), this, t, storedExecution, derivatives);
		}

		for (size_t t = 0; t < tCount; t++)
		{
			threads[t].join();
		}
		delete[] derivativeCalculators;

		NeuronConnectionsInfo* connectionsGradients = new NeuronConnectionsInfo[tCount];
		float** fieldsGradients = new float* [tCount];
		tuple<float, float> statesGradients(0.0f, 0.0f);
		for (size_t i = 0; i < tCount; i++)
		{

		}

		delete[] derivatives;
		delete[] threads;
		tuple<NeuronConnectionsInfo*, float**> output;
		return output;
	}

private:
	class DerivativeCalculator
	{
		void operator()(LSTMNeuron* neuron, size_t t, NeuronStoredValues* executionResults, NeuronStoredValues* derivatives)
		{
			derivatives[t] = neuron->GetDerivatives(executionResults[t]);
		}
	};

	/// <returns>
	/// tuple(previous hiddenStateGradient, previous cellStateGradient)
	/// </returns>
	tuple<float, float> CalculateGradients(size_t t, NeuronConnectionsInfo* connectionsGradients, float** fieldsGradients, NeuronStoredValues& derivatives,
		float* neuronCost, tuple<float, float> hiddenCellGradients, 
		float*** networkActivations, float*** networkCosts)
	{
		float currentCost = neuronCost[t];
		currentCost += get<0>(hiddenCellGradients);
		float outputWeightMultiplicationGradient = currentCost *= derivatives.OutputWeightMultiplication;

		currentCost *= derivatives.CellStateTanh;
		currentCost += get<1>(hiddenCellGradients);

		currentCost *= derivatives.StoreGateAddition;
		float storeGateGradient = currentCost;

		storeGateGradient *= derivatives.StoreGateMultiplication;
		float storeGateTanhWeightGradient = storeGateGradient * derivatives.StoreTanhWeightMultiplication;
		float storeGateSigmoidWeightGradient = storeGateGradient * derivatives.StoreSigmoidWeightMultiplication;

		currentCost *= derivatives.ForgetGateMultiplication;
		float previousCellStateCost = currentCost;
		float forgetGateGradient = currentCost;

		float forgetGateWeightGradient = forgetGateGradient *= derivatives.ForgetWeightMultiplication;


		currentCost = outputWeightMultiplicationGradient;

		currentCost *= derivatives.HiddenLinearSigmoid;
		currentCost *= connections.GetDerivative(networkActivations[t]);

		float previousHiddenStateCost = currentCost;

		float* weightsGradients = connections.GetGradients(currentCost, networkActivations[t], networkCosts[t]);
		connectionsGradients[t] = NeuronConnectionsInfo(connections.GetConnectionCount(), currentCost, weightsGradients, NULL, NULL);


		tuple<float, float> output(previousHiddenStateCost, previousCellStateCost);
		return output;
	}

	NeuronStoredValues GetDerivatives(NeuronStoredValues& executionResults)
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

	void DeleteMemory()
	{
		hiddenState = 0;
		cellState = 0;
	}
};

