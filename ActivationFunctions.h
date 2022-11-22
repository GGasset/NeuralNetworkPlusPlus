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
		return sinf(x) / cosf(x);
	}
};

