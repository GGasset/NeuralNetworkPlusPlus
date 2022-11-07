using namespace std;
#include "Neuron.h"
#include <list>
#include <tuple>

#pragma once
class NeuralNetwork
{
private:
	list<list<Neuron>> neurons;
	ActivationFunctions::ActivationFunction ActivationFunction;

public:
	NeuralNetwork(long shapeLength, long* shape, double bias, ActivationFunctions::ActivationFunction activationFunction, double minWeight, double weightClosestTo0, double maxWeight)
	{
		neurons = list<list<Neuron>>();
		for (long i = 1; i < shapeLength; i++)
		{
			list<Neuron> currentLayer = list<Neuron>();
			for (long j = 0; j < shape[i]; j++)
			{
				currentLayer.push_back(Neuron(i, shape[i - 1], bias, minWeight, weightClosestTo0, maxWeight));
			}
			neurons.push_back(currentLayer);
		}

		ActivationFunction = activationFunction;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="input"></param>
	/// <returns>tuple(networkLinears, networkActivations)</returns>
	tuple<double**, double**> ExecuteStore(double* input)
	{
		double** networkLinearFunctions = new double* [GetNetworkLayerCount()];
		double** networkActivations = new double* [GetNetworkLayerCount() + 1];
		networkActivations[0] = input;

		auto layerIter = neurons.begin();
		for (long i = 0; i < GetNetworkLayerCount(); i++, layerIter++)
		{
			tuple<double*, double*> layerExecutionResults = ExecuteStoreLayer(layerIter, networkActivations);
			networkLinearFunctions[i] = get<0>(layerExecutionResults);
			networkActivations[i + 1] = get<1>(layerExecutionResults);
		}
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="layerIter"></param>
	/// <param name="neuronActivations"></param>
	/// <returns>tuple(linearFunctions, neuronActivations)</returns>
	tuple<double*, double*> ExecuteStoreLayer(list<list<Neuron>>::iterator layerIter, double** neuronActivations)
	{
		list<Neuron> layer = (*layerIter);
		long layerLength = layer.size();
		double* layerLinears = new double[layerLength];
		double* layerActivations = new double[layerLength];

		auto neuronIter = layer.begin();
		for (long i = 0; neuronIter != layer.end(); i++, neuronIter++)
		{
			tuple<double, double> neuronExecutionResults = (*neuronIter).ExecuteStore(neuronActivations, ActivationFunction);
			layerLinears[i] = get<0>(neuronExecutionResults);
			layerActivations[i] = get<1>(neuronExecutionResults);
		}
		tuple<double*, double*> layerExecutionResults(layerLinears, layerActivations);
	}

	ActivationFunctions::ActivationFunction GetActivationFunction()
	{
		return ActivationFunction;
	}

	/// <summary>
	/// Output includes input layer
	/// </summary>
	/// <returns></returns>
	list<long> GetNetworkShape()
	{
		long networkLength = GetNetworkLayerCount();
		list<long> shape = list<long>();
		shape.push_back(GetNetworkInputLength());

		auto layerIterator = neurons.begin();
		for (long i = 0; layerIterator != neurons.end(); i++, layerIterator++)
		{
			shape.push_back((*layerIterator).size());
		}
		return shape;
	}

	/// <summary>
	/// Output doesn't include input layer
	/// </summary>
	/// <returns></returns>
	long GetNetworkLayerCount()
	{
		return neurons.size();
	}

	long GetNetworkInputLength()
	{
		auto layerIterator = neurons.begin();
		auto neuronIterator = (*layerIterator).begin();
		return (*neuronIterator).connections.GetConnectionCount();
	}
};

