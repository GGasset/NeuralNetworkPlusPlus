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
			for (long i = 0; i < shape[i]; i++)
			{
				currentLayer.push_back(Neuron(i, shape[i - 1], bias, minWeight, weightClosestTo0, maxWeight));
			}
			neurons.push_back(currentLayer);
		}

		ActivationFunction = activationFunction;
	}

	//tuple<double* ExecuteStore(double* input)

	ActivationFunctions::ActivationFunction GetActivationFunction()
	{
		return ActivationFunction;
	}

	list<long> GetNetworkShape()
	{
		long networkLength = GetNetworkLength();
		list<long> shape = list<long>();

		auto layerIterator = neurons.begin();
		for (long i = 0; layerIterator != neurons.end(); i++, layerIterator++)
		{
			shape.push_back((*layerIterator).size());
		}
		return shape;
	}

	long GetNetworkLength()
	{
		return neurons.size();
	}
};

