#include "INeuron.h"
#include "Neuron.h"
#include "LSTMNeuron.h"

#pragma once
class RNN
{
public:
	enum NeuronType
	{
		neuron,
		lstmNeuron
	};

private:
	INeuron*** neurons;
	ActivationFunctions::ActivationFunction activationFunction;
	size_t shapeLength;
	size_t* shape;

public:
	RNN(size_t shapeLength, size_t* shape, NeuronType* layerTypes, ActivationFunctions::ActivationFunction activationFunction,
		float bias, float minWeight, float weightClosestTo0, float maxWeight)
	{
		this->shapeLength = shapeLength;
		this->shape = shape;
		this->activationFunction = activationFunction;

		neurons = new INeuron **[shapeLength];
		for (size_t i = 1; i < shapeLength; i++)
		{
			neurons[i] = new INeuron *[shape[i]];
			for (size_t j = 0; j < shape[i]; j++)
			{
				INeuron* toAdd;
				switch (layerTypes[i - 1])
				{
				case neuron:
					toAdd = new Neuron(i, shape[i - 1], bias, minWeight, weightClosestTo0, maxWeight);
					break;
				case lstmNeuron:
					toAdd = new LSTMNeuron(i, shape[i - 1], bias, minWeight, weightClosestTo0, maxWeight);
					break;
				default:
					throw std::exception("NeuronType not implemented");
				}
				neurons[i][j] = toAdd;
			}
		}
	}
};

