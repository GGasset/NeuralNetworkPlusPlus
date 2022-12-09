#include "Derivatives.h"
#include <stdlib.h>
#include "INeuron.h"
using namespace std;

#pragma once
class Neuron : public INeuron
{
public:
	Neuron(size_t layerI, size_t previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
	}

	Neuron(float bias, size_t* connectionsX, size_t* connectionsY, float* weights)
	{
		connections.Bias = bias;
		connections.Xs = connectionsX;
		connections.Ys = connectionsY;
		connections.Weights = weights;
	}

	Neuron() {

	}

	float Execute(float** neuronActivations, ActivationFunctions::ActivationFunction activationType)
	{
		tuple<float,float> linearAndActivation = ExecuteStore(neuronActivations, activationType);
		float output = get<0>(linearAndActivation);
		return output;
	}

	NeuronStoredValues RecurrentExecuteStore(float** networkActivations, ActivationFunctions::ActivationFunction activationType)
	{
		tuple<float, float> storedValues = ExecuteStore(networkActivations, activationType);
		NeuronStoredValues output = NeuronStoredValues();
		output.LinearFunction = get<0>(storedValues);
		output.OutputActivation = get<1>(storedValues);
		return output;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="networkActivations"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(linear function, neuron activation)</returns>
	tuple<float, float> ExecuteStore(float** networkActivations, ActivationFunctions::ActivationFunction activationType)
	{
		float linearFunction = connections.LinearFunction(networkActivations);
		float activation = ActivationFunctions::Activate(linearFunction, activationType);

		tuple<float, float> output(linearFunction, activation);
		return output;
	}

	tuple<NeuronConnectionsInfo*, float**> GetRecurrentGradients(size_t tCount, NeuronStoredValues storedExecution, float* neuronCosts, float*** networkCosts, float*** networkActivations,
		ActivationFunctions::ActivationFunction activationType)
	{
		NeuronConnectionsInfo* gradients = new NeuronConnectionsInfo[tCount];
		std::thread* threads = new thread[tCount];
		for (size_t t = 0; t < tCount; t++)
		{
			threads[t] = thread(std::ref(gradients[t]), this, gradients, t, storedExecution, networkActivations, neuronCost, networkCosts, activationType);
		}

		for (size_t t = 0; t < tCount; t++)
		{
			threads[t].join();
		}

		tuple<NeuronConnectionsInfo*, float**> output(gradients, new float*[0]);
		return output;
	}

private:
	class GradientCalculator
	{
		void operator()(Neuron* neuron, NeuronConnectionsInfo* output, size_t outputI, NeuronStoredValues& storedExecution, float** networkActivations, float neuronCost, float** networkCosts,
						ActivationFunctions::ActivationFunction activationType)
		{
			tuple<float, float*> gradients = neuron->GetGradients(networkActivations, storedExecution.LinearFunction, neuronCost, networkCosts, activationType);
			output[outputI].Bias = get<0>(gradients);
			output[outputI].Weights = get<1>(gradients);
		}
	};

public:
	/// <summary>
	/// 
	/// </summary>
	/// <param name="linearFunction"></param>
	/// <param name="neuronCost"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(biasGradient, weightGradient)</returns>
	tuple<float, float*> GetGradients(float** networkActivations, float linearFunction, float neuronCost, float** networkCosts, ActivationFunctions::ActivationFunction activationType)
	{
		float biasGradient = neuronCost * Derivatives::DerivativeOf(linearFunction, activationType);
		float* weightGradients = connections.GetGradients(biasGradient, networkActivations, networkCosts);

		tuple<float, float*> output(biasGradient, weightGradients);
		return output;
	}

	void ApplyGradients(size_t tCount, NeuronConnectionsInfo* connectionsGradients, float** fieldsGradients, float learningRate)
	{
		for (size_t i = 0; i < tCount; i++)
		{
			connections.ApplyGradients(connectionsGradients[i], learningRate);
			delete[] fieldsGradients[i];
		}
		delete[] connectionsGradients;
		delete[] fieldsGradients;
	}

	void ApplyGradients(Neuron gradients, float learningRate)
	{
		connections.ApplyGradients(gradients.connections, learningRate);
	}
};

