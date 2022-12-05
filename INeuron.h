#include "NeuronConnectionsInfo.h"
#include "NeuronStoredValues.h"

#pragma once
class INeuron
{
public:
	NeuronConnectionsInfo connections;

	virtual float Execute(float** neuronActivations, ActivationFunctions::ActivationFunction activationType) = 0;

	virtual NeuronStoredValues RecurrentExecuteStore(float** networkActivations, ActivationFunctions::ActivationFunction activationType) = 0;
};

