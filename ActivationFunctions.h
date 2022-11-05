#pragma once
#include <math.h>
#include <cmath>

static class ActivationFunctions
{
public:
	static enum ActivationFunction
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
			throw new exception("Not implemented activation function used");
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

