#include "INeuron.h"
#include "Neuron.h"
#include "LSTMNeuron.h"

#pragma once
class RNN
{
public:
	enum NeuronType
	{
		Neuron,
		LSTMNeuron
	};

private:
	INeuron*** neurons;
	ActivationFunctions::ActivationFunction activationFunction;
	size_t shapeLength;
	int* shape;

public:
	RNN(size_t layerCount, int* shape, NeuronType* layerTypes, ActivationFunctions::ActivationFunction activationFunction)
	{

	}
};

