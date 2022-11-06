using namespace std;
#include "Neuron.h"
#include <list>

#pragma once
class NeuronalNetwork
{
private:
	list<list<Neuron>> neurons;
	ActivationFunctions::ActivationFunction ActivationFunction;

public:
	ActivationFunctions::ActivationFunction GetActivationFunction()
	{
		return ActivationFunction;
	}

	long* GetNetworkShape()
	{
		long networkLength = GetNetworkLength();
		long* shape = (long*)malloc(sizeof(long) * networkLength);
		if (!shape)
		{
			return GetNetworkShape();
		}

		auto layerIterator = neurons.cbegin();
		for (long i = 0; i < networkLength; i++, layerIterator++)
		{
			(shape[i]) = (*layerIterator).size();
		}
		return shape;
	}

	long GetNetworkLength()
	{
		return neurons.size();
	}
};

