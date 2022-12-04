#pragma once
#include <math.h>
#include <cmath>
#include <stdexcept>

class ActivationFunctions
{
public:
	enum ActivationFunction
	{
		None,
		RELU,
		Sigmoid,
		Tanh,
	};

	static float Activate(float x, ActivationFunction activationType)
	{
		switch (activationType)
		{
		case ActivationFunctions::RELU:
			return RELUActivation(x);
		case ActivationFunctions::Sigmoid:
			return SigmoidActivation(x);
		case Tanh:
			return TanhActivation(x);
		case None:
			return x;
		default:
			return NULL;
		}
	}

	static float RELUActivation(float x)
	{
		return fmaxf(0, x);
	}

	static float SigmoidActivation(float x)
	{
		return 1 / (1 + exp(-x));
	}

	static float TanhActivation(float x)
	{
		return (2 / (1 + exp(-2 * x))) - 1;
	}
};

