#include <cmath>

#pragma once
class Cost
{
public:
	enum CostFunction
	{
		SquaredMean,
	};

	static float GetCostOf(size_t networkOutputLength, float* neuronOutput, float* Y, CostFunction costType)
	{
		switch (costType)
		{
		case Cost::SquaredMean:
			return SquaredMeanLoss(networkOutputLength, neuronOutput, Y);
		default:
			return NULL;
		}
	}

	static float SquaredMeanLoss(size_t outputLength, float* neuronOutput, float* Y)
	{
		float mean = 0;
		for (size_t i = 0; i < outputLength; i++)
		{
			mean += SquaredMeanLoss(neuronOutput[i], Y[i]);
		}
	}

	static float SquaredMeanLoss(float neuronOutput, float Y)
	{
		return pow(neuronOutput - Y, 2);
	}
};

