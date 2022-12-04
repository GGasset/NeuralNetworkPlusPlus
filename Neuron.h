#include "NeuronConnectionsInfo.h"
#include "ActivationFunctions.h"
#include "Derivatives.h"
#include <stdlib.h>
#include <list>
#include <tuple>
using namespace std;

#pragma once
class Neuron
{
public:
	NeuronConnectionsInfo connections;

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
		tuple<float*> connectionsGradients = connections.GetGradients(biasGradient, networkActivations, networkCosts);

		float* weightGradients = get<0>(connectionsGradients);

		tuple<float, float*> output(biasGradient, weightGradients);
		return output;
	}

	void ApplyGradients(Neuron gradients, float learningRate)
	{
		connections.ApplyGradients(gradients.connections, learningRate);
	}
};

