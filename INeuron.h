#include "NeuronConnectionsInfo.h"
#include "NeuronStoredValues.h"
#include "ActivationFunctions.h"

#pragma once
class INeuron
{
public:
	NeuronConnectionsInfo connections;

	virtual float Execute(float** neuronActivations, ActivationFunctions::ActivationFunction activationType) = 0;

	virtual NeuronStoredValues RecurrentExecuteStore(float** networkActivations, ActivationFunctions::ActivationFunction activationType) = 0;

	virtual tuple<NeuronConnectionsInfo*, float**> GetRecurrentGradients(size_t tCount, NeuronStoredValues storedExecution, float* neuronCost, float*** networkCosts, float*** networkActivations,
																		ActivationFunctions::ActivationFunction activationType) = 0;

	virtual void ApplyGradients(size_t tCount, NeuronConnectionsInfo* connections, float** fieldsGradient, float learningRate) = 0;
};

