#include "Neuron.h"
#include "LSTMNeuron.h"

#pragma once
class NeuronHolder
{
public:
	enum NeuronType
	{
		LSTM,
		NonRecurrentNeuron
	};

	Neuron* Neuron;
	LSTMNeuron* LstmNeuron;
};

