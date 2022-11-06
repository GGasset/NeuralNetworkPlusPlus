#include <math.h>
#include <cmath>
#include <stdlib.h>
#include "ActivationFunctions.h"

#pragma once
class Derivatives
{
public:
	static double DerivativeOf(double linearFunction, ActivationFunctions::ActivationFunction ActivationType)
	{
		switch (ActivationType)
		{
		case ActivationFunctions::RELU:
			return RELUDerivative(linearFunction);
		case ActivationFunctions::Sigmoid:
			return SigmoidDerivative(linearFunction);
		default:
			return 0;
		}
	}

	static double RELUDerivative(double linearFunction)
	{
		return linearFunction * (linearFunction >= 0);
	}

	static double SigmoidDerivative(double linearFunction)
	{
		return ((-linearFunction - 1) * exp(-linearFunction - 1)) / pow(1 + exp(linearFunction), 2);
	}
};

