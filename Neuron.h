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

	Neuron(int layerI, int previousLayerLength, double bias, double minWeight, double weightClosestTo0, double maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerLength, bias, minWeight, weightClosestTo0, maxWeight);
	}

	Neuron(double bias, list<long> connectionsX, list<long> connectionsY, list<double> weights)
	{
		connections = NeuronConnectionsInfo(); 
		connections.Bias = bias;
		connections.Xs = connectionsX;
		connections.Ys = connectionsY;
		connections.Weights = weights;
	}

	Neuron() {

	}

	double Execute(double** neuronActivations, ActivationFunctions::ActivationFunction activationType)
	{
		tuple<double,double> linearAndActivation = ExecuteStore(neuronActivations, activationType);
		double output = get<0>(linearAndActivation);
		return output;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="networkActivations"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(linear function, neuron activation)</returns>
	tuple<double, double> ExecuteStore(double** networkActivations, ActivationFunctions::ActivationFunction activationType)
	{
		double linearFunction = connections.LinearFunction(networkActivations);
		double activation = ActivationFunctions::Activate(linearFunction, activationType);

		tuple<double, double> output(linearFunction, activation);
		return output;
	}


	/// <summary>
	/// 
	/// </summary>
	/// <param name="linearFunction"></param>
	/// <param name="NeuronCost"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(biasGradient, weightGradient, previousActivationGradients\)</returns>
	tuple<double, list<double>, list<double>> GetGradients(double** networkActivations, double linearFunction, double NeuronCost, ActivationFunctions::ActivationFunction activationType)
	{
		double biasGradient = NeuronCost * Derivatives::DerivativeOf(linearFunction, activationType);
		tuple<list<double>, list<double>> connectionsGradients = connections.GetGradients(biasGradient, networkActivations);

		list<double> weightGradients = get<0>(connectionsGradients);
		list<double> previousActivationsGradients = get<1>(connectionsGradients);

		tuple<double, list<double>, list<double>> output(biasGradient, weightGradients, previousActivationsGradients);
		return output;
	}

	void ApplyGradients(Neuron gradients)
	{
		connections.ApplyGradients(gradients.connections);
	}
};

