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

	static double Activate(double linearFunc, ActivationFunction activationType)
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

	static double RELUActivation(double linearFunc)
	{
		return fmax(0, linearFunc);
	}

	static double SigmoidActivation(double linearFunc)
	{
		return 1 / (1 + exp(-linearFunc));
	}
};

