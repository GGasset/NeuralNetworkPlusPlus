#pragma once
#include <math.h>
#include <cmath>
#include <stdexcept>

class ActivationFunctions
{
public:
	enum ActivationFunction
	{
		RELU,
		Sigmoid,
	};

	static float Activate(float linearFunc, ActivationFunction activationType)
	{
		switch (activationType)
		{
		case ActivationFunctions::RELU:
			return RELUActivation(linearFunc);
		case ActivationFunctions::Sigmoid:
			return SigmoidActivation(linearFunc);
		default:
			return NULL;
		}
	}

	static float RELUActivation(float linearFunc)
	{
		return fmax(0, linearFunc);
	}

	static float SigmoidActivation(float linearFunc)
	{
		return 1 / (1 + exp(-linearFunc));
	}
};

