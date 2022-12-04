#include <math.h>
#include <cmath>
#include <stdlib.h>
#include "ActivationFunctions.h"
#include "Cost.h"

#pragma once
class Derivatives
{
public:
	static float DerivativeOf(float linearFunction, ActivationFunctions::ActivationFunction ActivationType)
	{
		switch (ActivationType)
		{
		case ActivationFunctions::RELU:
			return RELUDerivative(linearFunction);
		case ActivationFunctions::Sigmoid:
			return SigmoidDerivative(linearFunction);
		case ActivationFunctions::None:
			return 1;
		default:
			return NULL;
		}
	}

	static float* DerivativeOf(size_t networkOutputLength, float* networkOutput, float* Y, Cost::CostFunction costFunction)
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

	static float RELUDerivative(float linearFunction)
	{
		return linearFunction * (linearFunction >= 0);
	}

	static float SigmoidDerivative(float linearFunction)
	{
		return ActivationFunctions::SigmoidActivation(linearFunction) * (1 - ActivationFunctions::SigmoidActivation(linearFunction));
	}


	//Cost derivatives

	static float* SquaredMeanDerivative(size_t networkOutputLength, float* networkOutput, float* Y)
	{
		float* output = new float[networkOutputLength];
		for (size_t i = 0; i < networkOutputLength; i++)
		{
			output[i] = SquaredMeanDerivative(networkOutput[i], Y[i]);
		}
		return output;
	}

	static float SquaredMeanDerivative(float neuronOutput, float Y)
	{
		return 2 * (neuronOutput - Y);
	}


	//Common derivatives

	static float MultiplicationDerivative(float a, float b, float Da, float Db)
	{
		return a * Da + b * Db;
	}

	static float DivisionDerivative(float a, float b, float Da, float Db)
	{
		return (Da * b - Db * a) / (b * b);
	}
};

