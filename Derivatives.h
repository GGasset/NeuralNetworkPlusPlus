#include <math.h>
#include <cmath>
#include <stdlib.h>
#include "ActivationFunctions.h"
#include "Cost.h"

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
			return NULL;
		}
	}

	static double* DerivativeOf(long networkOutputLength, double* networkOutput, double* Y, Cost::CostFunction costFunction)
	{
		switch (costFunction)
		{
		case Cost::SquaredMean:
			return SquaredMeanDerivative(networkOutputLength, networkOutput, Y);
		default:
			return NULL;
		}
	}

	// Activation function derivatives
	static double RELUDerivative(double linearFunction)
	{
		return linearFunction * (linearFunction >= 0);
	}

	static double SigmoidDerivative(double linearFunction)
	{
		return ActivationFunctions::SigmoidActivation(linearFunction) * (1 - ActivationFunctions::SigmoidActivation(linearFunction));
	}

	//Cost derivatives
	static double* SquaredMeanDerivative(long networkOutputLength, double* networkOutput, double* Y)
	{
		double* output = new double[networkOutputLength];
		for (long i = 0; i < networkOutputLength; i++)
		{
			output[i] = SquaredMeanDerivative(networkOutput[i], Y[i]);
		}
		return output;
	}

	static double SquaredMeanDerivative(double neuronOutput, double Y)
	{
		return 2 * (neuronOutput - Y);
	}
};

