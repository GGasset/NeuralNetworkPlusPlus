#include "NeuronConnectionsInfo.h";
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

	Neuron(int layerI, int previousLayerLength, float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
	}

	Neuron(float bias, list<long> connectionsX, list<long> connectionsY, list<float> weights)
	{
		connections = NeuronConnectionsInfo(); 
		connections.Bias = bias;
		connections.Xs = connectionsX;
		connections.Ys = connectionsY;
		connections.Weights = weights;
	}

	Neuron() {

	}

	double Execute(float** neuronActivations, ActivationFunctions::ActivationFunction activationType)
	{
		tuple<float,float> linearAndActivation = ExecuteStore(neuronActivations, activationType);
		double output = get<0>(linearAndActivation);
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
	/// <param name="NeuronCost"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(biasGradient, weightGradient, previousActivationGradients\)</returns>
	tuple<float, list<float>, list<float>> GetGradients(float** networkActivations, float linearFunction, float NeuronCost, ActivationFunctions::ActivationFunction activationType)
	{
		float biasGradient = NeuronCost * Derivatives::DerivativeOf(linearFunction, activationType);
		tuple<list<float>, list<float>> connectionsGradients = connections.GetGradients(biasGradient, networkActivations);

		list<float> weightGradients = get<0>(connectionsGradients);
		list<float> previousActivationsGradients = get<1>(connectionsGradients);

		tuple<float, list<float>, list<float>> output(biasGradient, weightGradients, previousActivationsGradients);
		return output;
	}

	void ApplyGradients(Neuron gradients, float learningRate)
	{
		connections.ApplyGradients(gradients.connections, learningRate);
	}
};

