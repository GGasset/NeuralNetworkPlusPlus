#include <math.h>
#include <cmath>
#include <stdlib.h>
#include "ActivationFunctions.h"
#include "Cost.h"

#pragma once
class Derivatives
{
public:
	static float DerivativeOf(float x, ActivationFunctions::ActivationFunction ActivationType)
	{
		switch (ActivationType)
		{
		case ActivationFunctions::RELU:
			return RELUDerivative(x);
		case ActivationFunctions::Sigmoid:
			return SigmoidDerivative(x);
		case ActivationFunctions::Tanh:
			return TanhDerivative(x);
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

	static float RELUDerivative(float x)
	{
		return x * (x >= 0);
	}

	static float SigmoidDerivative(float x)
	{
		return ActivationFunctions::SigmoidActivation(x) * (1 - ActivationFunctions::SigmoidActivation(x));
	}

	static float TanhDerivative(float x)
	{
		float bottomDerivative = expDerivative(-2 * x);
		return -DivisionDerivative(2, 1 + exp(-2 * x), 0, bottomDerivative);
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

	static float expDerivative(float x)
	{
		return exp(x);
	}
};

