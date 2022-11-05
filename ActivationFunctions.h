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

	static double RELU(double linearFunc)
	{
		return fmax(0, linearFunc);
	}

	static double Sigmoid(double linearFunc)
	{
		return 1 / (1 + exp(-linearFunc));
	}
};

