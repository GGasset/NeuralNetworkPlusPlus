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

	Neuron(int layerI, int previousLayerI, double bias, double minWeight, double weightClosestTo0, double maxWeight)
	{
		connections = NeuronConnectionsInfo(layerI, previousLayerI, bias, minWeight, weightClosestTo0, maxWeight);
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
	/// <returns>neuronActivation</returns>
	tuple<double, double> ExecuteStore(double** neuronActivations, ActivationFunctions::ActivationFunction activationType)
	{
		double linearFunction;
		double activation = ActivationFunctions::Activate(linearFunction = connections.Execute(neuronActivations), activationType);

		tuple<double, double> output(linearFunction, activation);
		return output;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="linearFunction"></param>
	/// <param name="NeuronCost"></param>
	/// <param name="activationType"></param>
	/// <returns>tuple(weightGradients, previousActivationGradients, biasGradient)</returns>
	tuple<list<double>, list<double>, double> GetGradients(double** networkActivations, double linearFunction, double NeuronCost, ActivationFunctions::ActivationFunction activationType)
	{
		double biasGradient = NeuronCost * Derivatives::DerivativeOf(linearFunction, activationType);
		tuple<list<double>, list<double>> connectionsGradients = connections.GetGradients(biasGradient, networkActivations);

		list<double> weightGradients = get<0>(connectionsGradients);
		list<double> previousActivationsGradients = get<1>(connectionsGradients);

		tuple<list<double>, list<double>, double> output(weightGradients, previousActivationsGradients, biasGradient);
		return output;
	}
};

