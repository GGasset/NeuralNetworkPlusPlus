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

	tuple<NeuronConnectionsInfo*, float**> GetRecurrentGradients(size_t tCount, NeuronStoredValues* storedExecution, float* neuronCosts, float*** networkCosts, float*** networkActivations,
		ActivationFunctions::ActivationFunction activationType)
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
		for (size_t t = 0; t < tCount; t++)
		{
			statesGradients = CalculateGradients(t, connectionsGradients, fieldsGradients[t], derivatives[t],
				neuronCosts, statesGradients,
				networkActivations, networkCosts);
		}

		delete[] derivatives;
		delete[] threads;
		tuple<NeuronConnectionsInfo*, float**> output(connectionsGradients, fieldsGradients);
		return output;
	}

private:
	/// <returns>
	/// tuple(previous hiddenStateGradient, previous cellStateGradient)
	/// </returns>
	tuple<float, float> CalculateGradients(size_t t, NeuronConnectionsInfo* connectionsGradients, float* fieldsGradients, NeuronStoredValues& derivatives,
		float* neuronCosts, tuple<float, float> hiddenCellGradients, 
		float*** networkActivations, float*** networkCosts)
	{
		float currentCost = neuronCosts[t];
		currentCost += get<0>(hiddenCellGradients);
		float outputWeightGradient = currentCost *= derivatives.OutputWeightMultiplication;

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


		currentCost = outputWeightGradient;

		currentCost *= derivatives.HiddenLinearSigmoid;
		currentCost *= connections.GetDerivative(networkActivations[t]);

		float previousHiddenStateCost = currentCost;

		float* weightsGradients = connections.GetGradients(currentCost, networkActivations[t], networkCosts[t]);
		connectionsGradients[t].Bias = currentCost;
		connectionsGradients[t].Weights = weightsGradients;

		fieldsGradients = new float[4];
		fieldsGradients[0] = forgetGateWeightGradient;
		fieldsGradients[1] = storeGateSigmoidWeightGradient;
		fieldsGradients[2] = storeGateTanhWeightGradient;
		fieldsGradients[3] = outputWeightGradient;

		tuple<float, float> output(previousHiddenStateCost, previousCellStateCost);
		return output;
	}

	class DerivativeCalculator
	{
		void operator()(LSTMNeuron* neuron, size_t t, NeuronStoredValues* executionResults, NeuronStoredValues* derivatives)
		{
			derivatives[t] = neuron->CalculateDerivatives(executionResults[t]);
		}
	};

	NeuronStoredValues CalculateDerivatives(NeuronStoredValues& executionResults)
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

public:
	void ApplyGradients(size_t tCount, NeuronConnectionsInfo* connectionsGradients, float** fieldsGradients, float learningRate)
	{
		std::thread* threads = new std::thread[tCount];
		GradientApplyer* applyers = new GradientApplyer[tCount];
		for (size_t t = 0; t < tCount; t++)
		{
			threads[t] = std::thread(std::ref(applyers[t]), connections, t, connectionsGradients, fieldsGradients, learningRate);
		}

		for (size_t t = 0; t < tCount; t++)
		{
			threads[t].join();
		}

		delete[] connectionsGradients;
		delete[] fieldsGradients;
	}

private:
	class GradientApplyer
	{
		void operator()(NeuronConnectionsInfo& connections, size_t t, NeuronConnectionsInfo* connectionsGradients, float** fieldsGradients, float learningRate)
		{
			connections.ApplyGradients(connectionsGradients[t], learningRate);
			delete[] fieldsGradients[t];
		}
	};

public:
	void DeleteMemory()
	{
		hiddenState = 0;
		cellState = 0;
	}
};

